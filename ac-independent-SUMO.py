"""
ac-independent-SUMO.py

Description:
    Implementation of actor critic adapted for multi-agent environments. This implementation is origianlly based on
    the Clean-RL version https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py but uses Cross-Entropy for the actor's
    loss computation. The critic utilizes the same Q-Network structure as as the other MARL methods in this repository and the
    actor utilizes the same structure as the critic but with the addition of a layer that utilizes the softmax activation function

Usage:
    python ac-indepndent-SUMO.py -c experiments/sumo-2x2-ac-independent.config    


References:
    - https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py 

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

# # TODO: fix conda environment to include the version of gym that has Monitor module
# from gym.wrappers import TimeLimit#, Monitor
from datetime import datetime
import random
import os
import csv
from pettingzoo.butterfly import pistonball_v6
from pettingzoo.mpe import simple_spread_v3

# SUMO dependencies
import sumo_rl
import sys
from sumo_utils.sumo_custom.sumo_custom_observation import CustomObservationFunction
from sumo_utils.sumo_custom.sumo_custom_reward import CreateSumoReward
from sumo_utils.sumo_custom.calculate_speed_control import CalculateSpeedError

# Config Parser
from marl_utils.MARLConfigParser import MARLConfigParser

from rl_core.actor_critic import QNetwork, Actor, one_hot_q_values
from marl_utils.linear_schedule import LinearSchedule
from marl_utils.replay_buffer import ReplayBuffer

if __name__ == "__main__":
    
    # Get config parameters                        
    parser = MARLConfigParser()
    args = parser.parse_args()

    # The SUMO environment is slightly different from the defaul PettingZoo envs so set a flag to indicate if the SUMO env is being used
    using_sumo = False  
    if args.gym_id == 'sumo':

        using_sumo = True

        # Make sure SUMO env variable is set
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("Please declare the environment variable 'SUMO_HOME'")

    if not args.seed:
        args.seed = int(datetime.now())

# TRY NOT TO MODIFY: setup the environment
if args.gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
experiment_time = str(datetime.now()).split('.')[0].replace(':','-')   
experiment_name = "{}__N{}__exp{}__seed{}__{}".format(args.gym_id, args.N, args.exp_name, args.seed, experiment_time)
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")

# Specify directories for logging 
nn_dir = f"nn/{experiment_name}"
csv_dir = f"csv/{experiment_name}"
os.makedirs(f"{nn_dir}/critic_networks")
os.makedirs(f"{nn_dir}/actor_networks")
os.makedirs(csv_dir)

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

# Define an additional output file for the sumo-specific data
if using_sumo:
    sumo_csv = "{}/_SUMO_alpha{}_gamma{}_{}".format(csv_dir, args.learning_rate, args.gamma, experiment_time)

print("\n=================== Environment Information ===================")
# Instantiate the environment 
if using_sumo:
    # Sumo must be created using the sumo-rl module
    # Note we have to use the parallel env here to conform to this implementation of dqn
    
    sumo_reward_function = CreateSumoReward(args=args)

    env = sumo_rl.parallel_env(net_file=args.net, 
                            route_file=args.route,
                            use_gui=args.sumo_gui,
                            max_green=args.max_green,
                            min_green=args.min_green,
                            num_seconds=args.sumo_seconds,
                            add_system_info=True,       # Default is True
                            add_per_agent_info=True,    # Default is True                                       
                            reward_fn=sumo_reward_function,
                            observation_class=CustomObservationFunction,
                            sumo_warnings=False)

else: 
    print(" > ENV ARGS: {}".format(args.env_args))
    exec(f" > env = {args.gym_id}.parallel_env({args.env_args})")


agents = env.possible_agents
print(" > agents:\n {}".format(agents))

num_agents = len(env.possible_agents)
print(" > num_agents:\n {}".format(num_agents))

# TODO: these dictionaries are deprecated, use action_space & observation_space functions instead
action_spaces = env.action_spaces
observation_spaces = env.observation_spaces


# CSV files to save episode metrics during training
with open(f"{csv_dir}/critic_loss.csv", "w", newline="") as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_loss', 'global_step'])
    csv_writer.writeheader()

with open(f"{csv_dir}/actor_loss.csv", "w", newline="") as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_actor_loss', 'global_step'])
    csv_writer.writeheader()    

# system_episode_reward: the cumulative reward of all agents during the episode
# global_step: the global step in training
with open(f"{csv_dir}/episode_reward.csv", "w", newline="") as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_reward', 'global_step'])
    csv_writer.writeheader()

# system_episode_max_speed: Maximum speed observed by all agents during an episode
# system_episode_min_max_speed: The lowest of all maximum speeds observed by all agents during an episode
#   i.e. if four agents observed max speeds of [6.6, 7.0, 10.0, 12.0] during the episode, 
#   system_episode_min_max_speed would return 6.6 and system_episode_max_speed would return 12.0
# system_accumulated_stopped: Accumulated number of stopped cars observed by all agents during the episode
with open(f"{csv_dir}/episode_max_speeds.csv", "w", newline="") as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_max_speed', 
                                                            'system_episode_min_max_speed', 
                                                            'system_accumulated_stopped', 
                                                            'global_step'])    
    csv_writer.writeheader()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.reset(seed=args.seed)

for agent in agents:
    action_spaces[agent].seed(args.seed)
    observation_spaces[agent].seed(args.seed)

# respect the default timelimit
# assert isinstance(env.action_space, Discrete), "only discrete action space is supported"
# TODO: Monitor was not working 
# if args.capture_video:
#     env = Monitor(env, f'videos/{experiment_name}')

# TRY NOT TO MODIFY: start the game
print(f" > Initializing environment")
obses, _ = env.reset()

# Initialize data structures for training
# NOTE: When using parameter sharing, we only need one network & optimizer but when not using parameter sharing,
# each agent gets its own
# Dictionary for storing replay buffers (maps agent to a replay buffer)
rb = {agent: ReplayBuffer(args.buffer_size) for agent in agents}
print(" > Initializing neural networks")

if args.parameter_sharing_model:
    # Parameter sharing is being used
    print(f"  > Parameter sharing enabled")
    eg_agent = agents[0]
    onehot_keys = {agent: i for i, agent in enumerate(agents)}

    if args.global_obs:
        print(f"   > Global observations enabled")
        # Define the observation space dimensions (depending on whether or not global observations are being used)
        observation_space_shape = tuple((shape+1) * (num_agents) for shape in observation_spaces[eg_agent].shape)

        global_obs = np.hstack(list(obses.values()))
        for agent in agents:
            onehot = np.zeros(num_agents)
            onehot[onehot_keys[agent]] = 1.0
            obses[agent] = np.hstack([onehot, global_obs])        

    else:
        print(f"   > Global observations NOT enabled")

        # Convert (X,) shape from tuple to int so it can be modified
        observation_space_shape = np.array(observation_spaces[eg_agent].shape).prod() + num_agents  
        observation_space_shape = tuple(np.array([observation_space_shape]))   
        
        for agent in agents:
            onehot = np.zeros(num_agents)
            onehot[onehot_keys[agent]] = 1.0
            obses[agent] = np.hstack([onehot, obses[agent]])

    # Single q-network (i.e. "critic") for training
    q_network = QNetwork(observation_space_shape, action_spaces[eg_agent].n).to(device)

    # The target q-network
    target_network = QNetwork(observation_space_shape, action_spaces[eg_agent].n).to(device)
    target_network.load_state_dict(q_network.state_dict())                              

    # Single actor network for training        
    actor_network = Actor(observation_space_shape, action_spaces[eg_agent].n).to(device)
    
    # Optimizer for the critic
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    # Optimizer for the actor
    actor_optimizer = optim.Adam(list(actor_network.parameters()), lr=args.learning_rate)
    
    print(f"  > Observation space shape: {observation_space_shape}".format(observation_space_shape))
    print(f"  > Actionspace shape: {action_spaces[agent].n}")    
    print(f"  > Q-network structure: { q_network}") 


else:
    print(f"  > Parameter sharing NOT enabled")
    # Dictionary for storing q-networks (maps agent to a q-network), these are the "critics"
    q_network = {}

    # Dictionary for storing target networks (maps agent to a network)
    target_network = {}
    
    # Dictionary for storing actor networks (maps agents to a network)
    actor_network = {}

    # Dictionary for storing optimizer for each agent's network
    optimizer = {}

    # Dictionary for storing the optimizers used to train the actor networks 
    actor_optimizer = {}

    for agent in agents:
        observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if args.global_obs else observation_spaces[agent].shape
        
        q_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
        
        target_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
        
        # Intialize the target network the same as the critic network
        target_network[agent].load_state_dict(q_network[agent].state_dict())

        actor_network[agent] = Actor(observation_space_shape, action_spaces[agent].n).to(device)
        
        # All agents use the same optimizer for training
        optimizer[agent] = optim.Adam(q_network[agent].parameters(), lr=args.learning_rate)
        
        actor_optimizer[agent] = optim.Adam(list(actor_network[agent].parameters()), lr=args.learning_rate)

        print(f"   > Agent: {agent}".format(agent))    
        print(f"   > Observation space shape: {observation_space_shape}".format(observation_space_shape))
        print(f"   > Action space shape: {action_spaces[agent].n}")

    # network of last agent
    print(f"  > Q-network structure: { q_network[agent]}")

    # Global states
    if args.global_obs:
        print(f"   > Global observations enabled")
        global_obs = np.hstack(list(obses.values()))
        obses = {agent: global_obs for agent in agents}

    else: 
        print(f"   > Global observations NOT enabled")



loss_fn = nn.MSELoss() # TODO: should the loss function be configurable?
actor_loss_fn = nn.CrossEntropyLoss()

print(" > Device: ",device.__repr__())


if args.render:
    env.render()    # TODO: verify that the sumo env supports render

# Dictionary that maps the each agent to its cumulative reward each episode
episode_rewards = {agent: 0 for agent in agents}

# Dictionary that maps each agent to the maximum speed observed at each step of the agent's episode
episode_max_speeds = {agent: [] for agent in agents}

# Dictionary that maps each agent to the avg speed reward obbtained by each agent
episode_avg_speed_rewards = {agent: 0 for agent in agents}

# Dictionary that maps each agent to the accumulated number of stopped cars it observes during episode
episode_accumulated_stopped = {agent: 0 for agent in agents}

# Dictionary that maps each agent to the action it selected
actions = {agent: None for agent in agents}

# Dictionary that maps each agent to the loss values for its critic network
losses = {agent: None for agent in agents}

# Dictionary that maps each agent to the loss values for its actor network
actor_losses = {agent: None for agent in agents}
lir_1 = 0
uir_1 = 0
var_1 = 0
cnt = 0

for global_step in range(args.total_timesteps):

    # ALGO LOGIC: put action logic here
    # Inflate/Deflate the total number of timesteps based on the exploration fraction 
    epsilon = LinearSchedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)

    # Set the action for each agent
    for agent in agents:
        if random.random() < epsilon:
            actions[agent] = action_spaces[agent].sample()
        else:
            # Actor choses the actions
            if args.parameter_sharing_model:
                action, _, _ = actor_network.get_action(obses[agent])
            else:
                action, _, _ = actor_network[agent].get_action(obses[agent])
            
            actions[agent] = action.detach().cpu().numpy()
            
            # Letting critic pick the actions
            # logits = q_network[agent].forward(obses[agent].reshape((1,)+obses[agent].shape))  # Used in SUMO but not in simple_spread
            # actions[agent] = torch.argmax(logits, dim=1).tolist()[0]

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obses, rewards, dones, _, _ = env.step(actions)

    if args.parameter_sharing_model:
        # When using parameter sharing, add one hot encoding for either global observations or independent observations
        if args.global_obs:
            global_next_obs = np.hstack(list(next_obses.values()))
            for agent in agents:
                onehot = np.zeros(num_agents)
                onehot[onehot_keys[agent]] = 1.0
                next_obses[agent] = np.hstack([onehot, global_next_obs])
        else:
            for agent in agents:
                onehot = np.zeros(num_agents)
                onehot[onehot_keys[agent]] = 1.0
                next_obses[agent] = np.hstack([onehot, next_obses[agent]])

    else:
        # Global states
        if args.global_obs:
            global_obs = np.hstack(list(next_obses.values()))
            next_obses = {agent: global_obs for agent in agents}

    if args.render:
        env.render()

    # Extract performance about how we're doing so far
    # Accumulated min reward received by any agent this step
    lir_1 += min(rewards.values())

    # Accumulated max reward received by any agent this step
    uir_1 += max(rewards.values())

    # Accumulated variance of rewards received by all agents this step
    var_1 += np.var(list(rewards.values()))
    cnt += 1

    for agent in agents:

        episode_rewards[agent] += rewards[agent]
        # TODO: need to modify this for global observations
        # max speed is the last element of the custom observation array
        episode_max_speeds[agent].append(next_obses[agent][-1])
        
        # Avg speed reward has been added to observation (as the second to last element)   
        agent_avg_speed = next_obses[agent][-2]

        # The wrapper class needs to be unwrapped for some reason in order to properly access info 
        info = env.unwrapped.env._compute_info()
        
        # Total number of cars stopped at this agent
        agent_cars_stopped = info[f'{agent}_stopped']
        
        if ((agent_cars_stopped == 0.0) and (agent_avg_speed == 0.0)):
            # No cars and no average speed means there are no cars present in the intersection
            avg_speed_reward = 0.0

        else:
            # Compute this metric only if there were cars present in the intersection 
            # This conforms to the way the avg speed rewards are calculated
            avg_speed_reward = CalculateSpeedError(speed=agent_avg_speed, 
                                            speed_limit=args.sumo_average_speed_limit,
                                            lower_speed_limit=args.sumo_average_speed_limit)
            
        episode_avg_speed_rewards[agent] += avg_speed_reward

        episode_accumulated_stopped[agent] += agent_cars_stopped

        rb[agent].put((obses[agent], actions[agent], rewards[agent], next_obses[agent], dones[agent]))

    # ALGO LOGIC: critic training
    if (global_step > args.learning_starts) and (global_step % args.train_frequency == 0):

        if args.parameter_sharing_model:
            # Randomly sample an agent to use to calculate the estimated state-action values
            # NOTE: Observations pulled from the replay buffer have one-hot encoding applied to them already
            agent = random.choice(agents)
            s_obses, s_actions, s_rewards, s_next_obses, s_dones = rb[agent].sample(args.batch_size)

            with torch.no_grad():
                target = torch.max(target_network.forward(s_next_obses), dim=1)[0]
                td_target = torch.Tensor(s_rewards).to(device) + args.gamma * target * (1 - torch.Tensor(s_dones).to(device))
            q_values = q_network.forward(s_obses)
            # Get the max Q(s,a) for each observation in the batch
            old_val = q_values.gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()

            loss = loss_fn(td_target, old_val)
            losses[agent] = loss.item()

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
            optimizer.step()

            # Actor training
            a, log_pi, action_probs = actor_network.get_action(s_obses)

            # Compute the loss for this agent's actor
            # NOTE: Actor uses cross-entropy loss function where
            # input is the policy dist and the target is the value function with one-hot encoding applied
            # Q-values from "critic" encoded so that the highest state-action value maps to a probability of 1
            q_values_one_hot = one_hot_q_values(q_values)    
            actor_loss = actor_loss_fn(action_probs, q_values_one_hot.to(device))
            actor_losses[agent] = actor_loss.item()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(list(actor_network.parameters()), args.max_grad_norm)
            actor_optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                # TODO: could also perform a "soft update" of the target network (see 
                # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#:~:text=%23%20Soft%20update%20of%20the%20target%20network%27s%20weights)
                target_network.load_state_dict(q_network.state_dict())

            if global_step % args.nn_save_freq == 0:
                for agent in agents:
                    torch.save(q_network.state_dict(), f"{nn_dir}/critic_networks/{global_step}.pt")
                    torch.save(actor_network.state_dict(), f"{nn_dir}/actor_networks/{global_step}.pt")

        else:
            # Update the networks for each agent
            for agent in agents:

                s_obses, s_actions, s_rewards, s_next_obses, s_dones = rb[agent].sample(args.batch_size)

                with torch.no_grad():
                    target_max = torch.max(target_network[agent].forward(s_next_obses), dim=1)[0]
                    td_target = torch.Tensor(s_rewards).to(device) + args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))
                q_values = q_network[agent].forward(s_obses)

                # Get the max Q(s,a) for each observation in the batch
                old_val = q_values.gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()

                # Compute loss for agent's critic
                loss = loss_fn(td_target, old_val)
                losses[agent] = loss.item()

                # optimize the model for the critic
                optimizer[agent].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(q_network[agent].parameters()), args.max_grad_norm)
                optimizer[agent].step()


                # Actor training
                a, log_pi, action_probs = actor_network[agent].get_action(s_obses)

                # Compute the loss for this agent's actor
                # NOTE: Actor uses cross-entropy loss function where
                # input is the policy dist and the target is the value function with one-hot encoding applied
                # Q-values from "critic" encoded so that the highest state-action value maps to a probability of 1
                q_values_one_hot = one_hot_q_values(q_values)    
                actor_loss = actor_loss_fn(action_probs, q_values_one_hot.to(device))
                actor_losses[agent] = actor_loss.item()

                actor_optimizer[agent].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(list(actor_network[agent].parameters()), args.max_grad_norm)
                actor_optimizer[agent].step()

                # update the target network
                if global_step % args.target_network_frequency == 0:
                    target_network[agent].load_state_dict(q_network[agent].state_dict())

                # Save loss values occasionally
                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss/" + agent, loss, global_step)
                    writer.add_scalar("losses/actor_loss/" + agent, actor_loss, global_step)

            # Save a snapshot of the actor and critic networks at this iteration of training
            if global_step % args.nn_save_freq == 0:
                for a in agents:
                    torch.save(q_network[a].state_dict(), f"{nn_dir}/critic_networks/{global_step}-{a}.pt")
                    torch.save(actor_network[a].state_dict(), f"{nn_dir}/actor_networks/{global_step}-{a}.pt")

        # Save loss values occassionally
        # NOTE: When using parameter sharing, loss is not calculated for all agents each update step. This means that when loss is logged, 
        # only 1 agent has the most updated. In other words, if there are 4 agents, the row in the loss csv file for training step X
        # will have values for all 4 agents but only one of those values will be current (the other 3 will be stale by some number of 
        # training steps). This really should not matter as the log file should still show the same trends for each agent and the entire
        # system overall
        if (global_step > args.learning_starts) and (global_step % args.train_frequency == 0):
            if (global_step % 100 == 0):
                # Log the data to TensorBoard
                system_loss = sum(list(losses.values()))
                writer.add_scalar("losses/system_td_loss/", system_loss, global_step)
                system_actor_loss = sum(list(actor_losses.values()))
                writer.add_scalar("losses/system_actor_loss/", system_actor_loss, global_step)

                # Log data to CSV
                with open(f"{csv_dir}/critic_loss.csv", "a", newline="") as csvfile:
                    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_loss', 'global_step'])
                    csv_writer.writerow({**losses, **{'system_loss': system_loss, 'global_step': global_step}})
                with open(f"{csv_dir}/actor_loss.csv", "a", newline="") as csvfile:    
                    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_actor_loss', 'global_step'])                        
                    csv_writer.writerow({**actor_losses, **{'system_actor_loss': system_actor_loss, 'global_step': global_step}})

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook 
    obses = next_obses

    # If all agents are done, log the results and reset the evnironment to continue training
    if np.prod(list(dones.values())) or (global_step % args.max_cycles == args.max_cycles-1): 
        # Accumulated reward of all agents
        system_episode_reward = sum(list(episode_rewards.values()))

        # Accumulated avg speed reward of all agents
        system_episode_avg_speed_reward = sum(list(episode_avg_speed_rewards.values()))
        
        # Accumulated number of cars stopped at each step for all agents        
        system_accumulated_stopped = sum(list(episode_accumulated_stopped.values()))

        # Calculate the maximum of all max speeds observed from each agent during the episode
        agent_max_speeds = {agent:0 for agent in agents}
        for agent in agents:
            agent_max_speeds[agent] = max(episode_max_speeds[agent])
        system_episode_max_speed = max(list(agent_max_speeds.values()))
        system_episode_min_max_speed = min(list(agent_max_speeds.values()))
        
        # The wrapper class needs to be unwrapped in order to properly access info 
        info = env.unwrapped.env._compute_info()
        
        # Total number of cars stopped at end of episode
        agents_total_stopped = info['agents_total_stopped']
        
        print(f"\n > global_step={global_step}")
        print(f"   > agent_max_speeds {agent_max_speeds}")
        print(f"   > system_episode_max_speed {system_episode_max_speed}")
        print(f"   > system_episode_min_max_speed {system_episode_min_max_speed}")
        print(f"   > system total avg speed reward: {system_episode_avg_speed_reward}")
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        print(f"   > system reward: {system_episode_reward} using definition: {args.sumo_reward}")
        print(f"   > total cars stopped at end of episode: {agents_total_stopped}")
        diff_1 = uir_1-lir_1
        # var_1 = var_1/(cnt-1e-7)
        lir_2 = min(episode_rewards.values())
        uir_2 = max(episode_rewards.values())
        diff_2 = uir_2-lir_2
        var_2 = np.var(list(episode_rewards.values())) 
        
        print(f"   > system_episode_diff_1={diff_1}")
        print(f"   > uir1={uir_1}")
        print(f"   > lir1={lir_1}")
        print(f"   > system_variance1={var_1}")
        print(f"   > system_episode_diff_2={diff_2}")
        print(f"   > uir2={uir_2}")
        print(f"   > lir2={lir_2}")
        print(f"   > system_variance2={var_2}")

        # Logging should only be done after we've started training, up until then, the agents are just getting experience
        if global_step > args.learning_starts:
            for agent in agents:
                writer.add_scalar("charts/episode_reward/" + agent, episode_rewards[agent], global_step)
            writer.add_scalar("charts/episode_reward/uir_1", uir_1, global_step)
            writer.add_scalar("charts/episode_reward/lir_1", lir_1, global_step)
            writer.add_scalar("charts/episode_reward/diff_1", diff_1, global_step)
            writer.add_scalar("charts/episode_reward/var_1", var_1, global_step)

            writer.add_scalar("charts/episode_reward/uir_2", uir_2, global_step)
            writer.add_scalar("charts/episode_reward/lir_2", lir_2, global_step)
            writer.add_scalar("charts/episode_reward/diff_2", diff_2, global_step)
            writer.add_scalar("charts/episode_reward/var_2", var_2, global_step)

            writer.add_scalar("charts/epsilon/", epsilon, global_step)
            writer.add_scalar("charts/system_episode_reward/", system_episode_reward, global_step)

            with open(f"{csv_dir}/episode_reward.csv", "a", newline="") as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_reward', 'global_step'])
                csv_writer.writerow({**episode_rewards, **{'system_episode_reward': system_episode_reward, 'global_step': global_step}})

            with open(f"{csv_dir}/episode_max_speeds.csv", "a", newline="") as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_max_speed', 'system_episode_min_max_speed', 'system_accumulated_stopped', 'global_step'])
                csv_writer.writerow({**agent_max_speeds, **{'system_episode_max_speed': system_episode_max_speed,
                                                            'system_episode_min_max_speed': system_episode_min_max_speed,
                                                            'system_accumulated_stopped' : system_accumulated_stopped,
                                                            'global_step': global_step}})

            # If we're using the SUMO env, also save some data specific to that environment
            if using_sumo:
                env.unwrapped.save_csv(sumo_csv, global_step)
            
        # Reset the env to continue training            
        obses, _ = env.reset()
        lir_1 = 0
        uir_1 = 0
        var_1 = 0
        cnt = 0

        if args.parameter_sharing_model:
            # Add one hot encoding for either global observations or independent observations once the environment has been reset
            if args.global_obs:
                global_obs = np.hstack(list(obses.values()))
                for agent in agents:
                    onehot = np.zeros(num_agents)
                    onehot[onehot_keys[agent]] = 1.0
                    obses[agent] = np.hstack([onehot, global_obs])
            else:
                for agent in agents:
                    onehot = np.zeros(num_agents)
                    onehot[onehot_keys[agent]] = 1.0
                    obses[agent] = np.hstack([onehot, obses[agent]])

        else:
            # Global states
            if args.global_obs:
                global_obs = np.hstack(list(obses.values()))
                obses = {agent: global_obs for agent in agents}

        if args.render:
            env.render()

        # Reset dictionaries for next episode
        episode_rewards = {agent: 0 for agent in agents}
        episode_avg_speed_rewards = {agent: 0 for agent in agents}
        episode_accumulated_stopped = {agent: 0 for agent in agents}
        episode_max_speeds = {agent: [0] for agent in agents} 
        actions = {agent: None for agent in agents}


env.close()
writer.close()

"""
dqn-independent-SUMO.py

Description:
    Implementation of independent Q-Learning to be used on various environments from the PettingZoo 
    library. This file was modified from dqn-independent.py to support use of the sumo-rl traffic simulator library
    https://github.com/LucasAlegre/sumo-rl which is not technically part of the PettingZoo module but 
    conforms to the Petting Zoo API. Configuration of this script is performed through a configuration file, 
    examples of which can be found in the experiments/ directory.

    Note that experiments using the SUMO traffic simulator also require 'net' and 'route' files to configure 
    the environment.

Usage:
    python dqn-indepndent-SUMO.py -c experiments/sumo-4x4-dqn-independent.config    

References:
    - https://github.com/LucasAlegre/sumo-rl 
    - https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf 

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import configargparse
from distutils.util import strtobool
import collections
import numpy as np
# import gym
# TODO: fix conda environment to include the version of gym that has Monitor module
# from gym.wrappers import TimeLimit#, Monitor
from datetime import datetime
import random
import os
import csv

# SUMO dependencies
import sumo_rl
import sys
from sumo_custom_observation import CustomObservationFunction
from sumo_custom_reward import MaxSpeedRewardFunction

# Config Parser
from MARLConfigParser import MARLConfigParser


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

    if (args.sumo_reward == "custom"):
        # Use the custom "max speed" reward function
        print ( " > Using CUSTOM reward")
        env = sumo_rl.parallel_env(net_file=args.net, 
                                route_file=args.route,
                                use_gui=args.sumo_gui,
                                max_green=args.max_green,
                                min_green=args.min_green,
                                num_seconds=args.sumo_seconds,
                                reward_fn=MaxSpeedRewardFunction,
                                observation_class=CustomObservationFunction,
                                sumo_warnings=False)
    else:
        print ( " > Using standard reward")
        # The 'queue' reward is being used here which returns the (negative) total number of vehicles stopped at all intersections
        env = sumo_rl.parallel_env(net_file=args.net, 
                                route_file=args.route,
                                use_gui=args.sumo_gui,
                                max_green=args.max_green,
                                min_green=args.min_green,
                                num_seconds=args.sumo_seconds,
                                reward_fn=args.sumo_reward,
                                observation_class=CustomObservationFunction,
                                sumo_warnings=False)

else: 
    print(" > ENV ARGS: {}".format(args.env_args))
    exec(f"env = {args.gym_id}.parallel_env({args.env_args})")

agents = env.possible_agents
num_agents = len(env.possible_agents)
# TODO: these dictionaries are deprecated, use action_space & observation_space functions instead
action_spaces = env.action_spaces
observation_spaces = env.observation_spaces

agents = env.possible_agents
print(" > agents:\n {}".format(agents))

num_agents = len(env.possible_agents)
print(" > num_agents:\n {}".format(num_agents))

# TODO: these dictionaries are deprecated, use action_space & observation_space functions instead
action_spaces = env.action_spaces
print(" > action_spaces:\n {}".format(action_spaces))

observation_spaces = env.observation_spaces
print(" > observation_spaces:\n {}".format(observation_spaces))

# CSV files to save episode metrics during training
# system_episode_reward: the cumulative reward of all agents during the episode
# global_step: the global step in training
with open(f"{csv_dir}/td_loss.csv", "w", newline="") as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_loss', 'global_step'])
    csv_writer.writeheader()
with open(f"{csv_dir}/episode_reward.csv", "w", newline="") as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_reward', 'global_step'])
    csv_writer.writeheader()

# system_episode_max_speed: Maximum speed observed by all agents during an episode
# system_episode_min_max_speed: The lowest of all maximum speeds observed by all agents during an episode
#   i.e. if four agents observed max speeds of [6.6, 7.0, 10.0, 12.0] during the episode, 
#   system_episode_min_max_speed would return 6.6 and system_episode_max_speed would return 12.0
with open(f"{csv_dir}/episode_max_speeds.csv", "w", newline="") as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_max_speed', 'system_episode_min_max_speed', 'global_step'])    
    csv_writer.writeheader()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.reset(seed=args.seed)
# env.action_space.seed(args.seed)
# env.observation_space.seed(args.seed)
for agent in agents:
    action_spaces[agent].seed(args.seed)
    observation_spaces[agent].seed(args.seed)
    # assert isinstance(action_spaces[agent], Discrete), "only discrete action space is supported"
# respect the default timelimit
# assert isinstance(env.action_space, Discrete), "only discrete action space is supported"
# TODO: Monitor was not working 
# if args.capture_video:
#     env = Monitor(env, f'videos/{experiment_name}')

# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, observation_space_shape, action_space_dim):
        super(QNetwork, self).__init__()
        hidden_size = 64    # TODO: should we make this a config parameter?
        self.fc1 = nn.Linear(np.array(observation_space_shape).prod(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space_dim)

    def forward(self, x):
        x = torch.Tensor(x).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    '''
    Defines a schedule for decaying epsilon during the training procedure
    '''
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

# Initialize data structures for training
rb = {} # Dictionary for storing replay buffers (maps agent to a replay buffer)
q_network = {}  # Dictionary for storing q-networks (maps agent to a q-network)
target_network = {} # Dictionary for storing target networks (maps agent to a network)
optimizer = {}  # Dictionary for storing optimizers for each RL problem

for agent in agents:
    observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if args.global_obs else observation_spaces[agent].shape
    rb[agent] = ReplayBuffer(args.buffer_size)
    q_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
    target_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
    target_network[agent].load_state_dict(q_network[agent].state_dict())    # Intialize the target network the same as the main network
    optimizer[agent] = optim.Adam(q_network[agent].parameters(), lr=args.learning_rate) # All agents use the same optimizer for training

loss_fn = nn.MSELoss() # TODO: should the loss function be configurable?
print(device.__repr__())
print(q_network[agent]) # network of last agent

# TRY NOT TO MODIFY: start the game
obses, _ = env.reset()

# Global states
if args.global_obs:
    global_obs = np.hstack(list(obses.values()))
    obses = {agent: global_obs for agent in agents}

if args.render:
    env.render()    # TODO: verify that the sumo env supports render

episode_rewards = {agent: 0 for agent in agents}        # Dictionary that maps the each agent to its cumulative reward each episode
episode_max_speeds = {agent: [] for agent in agents}    # Dictionary that maps each agent to the maximum speed observed at each step of the agent's episode
actions = {agent: None for agent in agents}             # Dictionary that maps each agent to the action it selected
losses = {agent: None for agent in agents}              # Dictionary that maps each agent to the loss values for its critic network
lir_1 = 0
uir_1 = 0
var_1 = 0
cnt = 0

for global_step in range(args.total_timesteps):

    # ALGO LOGIC: put action logic here
    epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)

    # Set the action for each agent
    for agent in agents:
        if random.random() < epsilon:
            actions[agent] = action_spaces[agent].sample()
        else:
            logits = q_network[agent].forward(obses[agent].reshape((1,)+obses[agent].shape))
            actions[agent] = torch.argmax(logits, dim=1).tolist()[0]

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obses, rewards, dones, _, _ = env.step(actions)

    # Global states
    if args.global_obs:
        global_obs = np.hstack(list(next_obses.values()))
        next_obses = {agent: global_obs for agent in agents}

    if args.render:
        env.render()

    # Extract performance about how we're doing so far
    lir_1 += min(rewards.values())          # Accumulated min reward received by any agent this step
    uir_1 += max(rewards.values())          # Accumulated max reward received by any agent this step
    var_1 += np.var(list(rewards.values())) # Accumulated variance of rewards received by all agents this step
    cnt += 1

    # Update the networks for each agent
    for agent in agents:

        episode_rewards[agent] += rewards[agent]
        # TODO: need to modify this for global observations
        episode_max_speeds[agent].append(next_obses[agent][-1]) # max speed is the last element of the custom observation array

        # ALGO LOGIC: training.
        rb[agent].put((obses[agent], actions[agent], rewards[agent], next_obses[agent], dones[agent]))
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            s_obses, s_actions, s_rewards, s_next_obses, s_dones = rb[agent].sample(args.batch_size)
            with torch.no_grad():
                
                target_max = torch.max(target_network[agent].forward(s_next_obses), dim=1)[0]
                td_target = torch.Tensor(s_rewards).to(device) + args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))
            old_val = q_network[agent].forward(s_obses).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
            loss = loss_fn(td_target, old_val)
            losses[agent] = loss.item()

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss/" + agent, loss, global_step)

            # optimize the model
            optimizer[agent].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(q_network[agent].parameters()), args.max_grad_norm)
            optimizer[agent].step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network[agent].load_state_dict(q_network[agent].state_dict())

        if global_step % args.nn_save_freq == 0:
            for a in agents:
                torch.save(q_network[a].state_dict(), f"{nn_dir}/{global_step}-{a}.pt")

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook 
    obses = next_obses

    if global_step > args.learning_starts and global_step % args.train_frequency == 0:
        if global_step % 100 == 0:
            system_loss = sum(list(losses.values()))
            writer.add_scalar("losses/system_td_loss/", system_loss, global_step)

            with open(f"{csv_dir}/td_loss.csv", "a", newline="") as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_loss', 'global_step'])
                csv_writer.writerow({**losses, **{'system_loss': system_loss, 'global_step': global_step}})

    # If all agents are done, log the results and reset the evnironment to continue training
    if np.prod(list(dones.values())) or global_step % args.max_cycles == args.max_cycles-1: 
        system_episode_reward = sum(list(episode_rewards.values())) # Accumulated reward of all agents

        # Calculate the maximum of all max speeds observed from each agent during the episode
        agent_max_speeds = {agent:0 for agent in agents}
        for agent in agents:
            agent_max_speeds[agent] = max(episode_max_speeds[agent])
        system_episode_max_speed = max(list(agent_max_speeds.values()))
        system_episode_min_max_speed = min(list(agent_max_speeds.values()))
        print(" >>> agent_max_speeds {}".format(agent_max_speeds))
        print(" >>> system_episode_max_speed {}".format(system_episode_max_speed))
        print(" >>> system_episode_min_max_speed {}".format(system_episode_min_max_speed))

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        print(f" >>> global_step={global_step}, system_episode_reward={system_episode_reward}")
        diff_1 = uir_1-lir_1
        # var_1 = var_1/(cnt-1e-7)
        lir_2 = min(episode_rewards.values())
        uir_2 = max(episode_rewards.values())
        diff_2 = uir_2-lir_2
        var_2 = np.var(list(episode_rewards.values())) 
        
        print(f" >>> system_episode_diff_1={diff_1}")
        print(f" >>> uir1={uir_1}")
        print(f" >>> lir1={lir_1}")
        print(f" >>> system_variance1={var_1}")
        print(f" >>> system_episode_diff_2={diff_2}")
        print(f" >>> uir2={uir_2}")
        print(f" >>> lir2={lir_2}")
        print(f" >>> system_variance2={var_2}")

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
                csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_max_speed', 'system_episode_min_max_speed', 'global_step'])
                csv_writer.writerow({**agent_max_speeds, **{'system_episode_max_speed': system_episode_max_speed,
                                                            'system_episode_min_max_speed': system_episode_min_max_speed,
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

        # Global states
        if args.global_obs:
            global_obs = np.hstack(list(obses.values()))
            obses = {agent: global_obs for agent in agents}

        if args.render:
            env.render()
        episode_rewards = {agent: 0 for agent in agents}
        episode_max_speeds = {agent: [0] for agent in agents} 
        actions = {agent: None for agent in agents}


env.close()
writer.close()

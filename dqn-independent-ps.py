"""
dqn-independent-ps.py

Description:
    Implementation of independent Q-Learning with parameter sharing to be used on various environments from the PettingZoo 
    library. This file was modified from to support use of the sumo-rl traffic simulator library
    https://github.com/LucasAlegre/sumo-rl which is not technically part of the PettingZoo module but 
    conforms to the Petting Zoo API. Configuration of this script is performed through a configuration file, 
    examples of which can be found in the experiments/ directory.

    Note that experiments using the SUMO traffic simulator also require 'net' and 'route' files to configure 
    the environment.

Usage:
    python dqn-indepndent-ps.py -c experiments/sumo-4x4-dqn-independent-ps.config    

References:
    - https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf 

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from distutils.util import strtobool
import collections
import numpy as np

# TODO: fix conda environment to include the version of gym that has Monitor module
from datetime import datetime
import random
import os
import csv

# SUMO dependencies
import sumo_rl
import sys
from sumo_custom_observation import CustomObservationFunction
from sumo_custom_reward import MaxSpeedRewardFunction
from linear_schedule import LinearSchedule
from actor_critic import QNetwork

# Config Parser
from MARLConfigParser import MARLConfigParser

if __name__ == "__main__":

    # Get config parameters                        
    parser = MARLConfigParser()
    args = parser.parse_args()

    # The SUMO environment is slightly different from the defaul PettingZoo envs so set a flag to indicate if the SUMO env is being used
    args = parser.parse_args()
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


def one_hot(a, size):
    b = np.zeros((size))
    b[a] = 1
    return b


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
os.makedirs(nn_dir)
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
    exec(f"import pettingzoo.{args.gym_id}") # lol
    exec(f"env = pettingzoo.{args.gym_id}.parallel_env(N={args.N}, local_ratio=0.5, max_cycles={args.max_cycles}, continuous_actions=False)") # lol

agents = env.possible_agents
print(" > agents:\n {}".format(agents))

num_agents = len(env.possible_agents)
print(" > num_agents:\n {}".format(num_agents))

# TODO: these dictionaries are deprecated, use action_space & observation_space functions instead
action_spaces = env.action_spaces
print(" > action_spaces:\n {}".format(action_spaces))

observation_spaces = env.observation_spaces
print(" > observation_spaces:\n {}".format(observation_spaces))

onehot_keys = {agent: i for i, agent in enumerate(agents)}
print(" > onehot_keys:\n {}".format(onehot_keys))

# TODO: Plotting loss is currently disabled for this implementation - see below
with open(f"{csv_dir}/td_loss.csv", "w", newline="") as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_loss', 'global_step'])
    csv_writer.writeheader()
with open(f"{csv_dir}/episode_reward.csv", "w", newline="") as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_reward', 'global_step'])
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

# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    # TODO: need to understand difference between sample and get here, the both appear to provide 
    # random experience tuples
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

    def get(self, mini_batch_indices):
        mini_batch = []
        for i in mini_batch_indices:
            mini_batch.append(self.buffer[i])
        
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


# Initialize data structures for training
eg_agent = agents[0]

# Define the shape of the observation space depending on if we're using a global observation or not
# Regardless, we need to add an array of length num_agents to the observation to account for one hot encoding
if args.global_obs:
    observation_space_shape = tuple((shape+1) * (num_agents) for shape in observation_spaces[eg_agent].shape)
else:
    observation_space_shape = np.array(observation_spaces[eg_agent].shape).prod() + num_agents  # Convert (X,) shape from tuple to int so it can be modified
    observation_space_shape = tuple(np.array([observation_space_shape]))                        # Convert int to array and then to a tuple
 

rb = {} # Dictionary for storing replay buffers (maps agent to a replay buffer)

print(" > INITIALIZING NEURAL NETWORKS")
for agent in agents:
    rb[agent] = ReplayBuffer(args.buffer_size)
q_network = QNetwork(observation_space_shape, action_spaces[eg_agent].n).to(device) # In parameter sharing, all agents utilize the same q-network
target_network = QNetwork(observation_space_shape, action_spaces[eg_agent].n).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)   

loss_fn = nn.MSELoss()  # TODO: should the loss function be configurable?

print(" > Device: ",device.__repr__())
print(" > Q_network structure: ", q_network) # network of last agent

# TRY NOT TO MODIFY: start the game
obses, _ = env.reset()


# Add one hot encoding for either global observations or independent observations
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

if args.render:
    env.render()    # TODO: verify that the sumo env supports render

episode_rewards = {agent: 0 for agent in agents}
actions = {agent: None for agent in agents}
losses = {agent: None for agent in agents}  # TODO: Unsure if it makes sense to store a loss for each agent in independt DQN with PS

lir_1 = 0
uir_1 = 0
var_1 = 0
cnt = 0
num_turns = 1

for global_step in range(args.total_timesteps):

    # ALGO LOGIC: put action logic here
    epsilon = LinearSchedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)

    for agent in agents:
        if random.random() < epsilon:
            actions[agent] = action_spaces[agent].sample()
        else:
            logits = q_network.forward(obses[agent].reshape((1,)+obses[agent].shape))
            actions[agent] = torch.argmax(logits, dim=1).tolist()[0]

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obses, rewards, dones, _, _ = env.step(actions)

    # Add one hot encoding for either global observations or independent observations
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

    if args.render:
        env.render()

    lir_1 += min(rewards.values())          # Accumulated min reward received by any agent this step
    uir_1 += max(rewards.values())          # Accumulated max reward received by any agent this step
    var_1 += np.var(list(rewards.values())) # Accumulated variance of rewards received by all agents this step
    cnt += 1
    for agent in agents:
        
        episode_rewards[agent] += rewards[agent]

        rb[agent].put((obses[agent], actions[agent], rewards[agent], next_obses[agent], dones[agent]))
    
    # ALGO LOGIC: training
    # In DQN without parameter sharing, each agent's network is updated independently
    # Experience from that agent is used to estimate the state-action value function for that agent but in parameter sharing, the state-action 
    # value function is estimated using the exeprience from a random agent
    if global_step > args.learning_starts and global_step % args.train_frequency == 0:
        agent = random.choice(agents) 
        # turn = int(global_step/num_turns)%num_agents    # Pick the agent around which the minibatch will be centered
        # agent = agents[turn]
        # TODO: why do we need dictionaries here? we're only using the experience from the random agent
        sample_batch_indices = np.random.randint(low=0, high=len(rb[agent].buffer), size=args.batch_size)
        s_obses = {}
        s_actions = {}
        s_rewards = {}
        s_next_obses = {}
        s_dones = {}
        for a in agents:
            s_obses[a], s_actions[a], s_rewards[a], s_next_obses[a], s_dones[a] = rb[a].get(sample_batch_indices)
        with torch.no_grad():
            target_maxes = []
            target = torch.max(target_network.forward(s_next_obses[agent]), dim=1)[0]
            td_target = torch.Tensor(s_rewards[agent]).to(device) + args.gamma * target * (1 - torch.Tensor(s_dones[agent]).to(device))
        old_val = q_network.forward(s_obses[agent]).gather(1, torch.LongTensor(s_actions[agent]).view(-1,1).to(device)).squeeze()
        loss = loss_fn(td_target, old_val)
        losses[agent] = loss.item()

        # optimize the model
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
        optimizer.step()

        # update the target network
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

        if global_step % args.nn_save_freq == 0:
            for agent in agents:
                torch.save(q_network.state_dict(), f"{nn_dir}/{global_step}.pt")

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
        system_episode_reward = sum(list(episode_rewards.values()))  # Accumulated reward of all agents

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        print(f"global_step={global_step}, system_episode_reward={system_episode_reward}")
        diff_1 = uir_1-lir_1
        var_1 = var_1/(cnt-1e-7)
        lir_2 = min(episode_rewards.values())
        uir_2 = max(episode_rewards.values())
        diff_2 = uir_2-lir_2
        var_2 = np.var(list(episode_rewards.values()))
        print(f"system_episode_diff_1={diff_1}")
        print(f"uir1={uir_1}")
        print(f"lir1={lir_1}")
        print(f"system_variance1={var_1}")
        print(f"system_episode_diff_2={diff_2}")
        print(f"uir2={uir_2}")
        print(f"lir2={lir_2}")
        print(f"system_variance2={var_2}")

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

            # If we're using the SUMO env, also save some data specific to that environment
            if using_sumo:
                env.unwrapped.save_csv(sumo_csv, global_step)

        # Reset environment and various metrics since the episode completed
        obses, _ = env.reset()
        lir_1 = 0
        uir_1 = 0
        var_1 = 0
        cnt = 0

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

        if args.render:
            env.render()
        episode_rewards = {agent: 0 for agent in agents}
        actions = {agent: None for agent in agents}


env.close()
writer.close()

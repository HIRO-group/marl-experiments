"""
dqn-greedy.py

Description:
    Implementation of distributed "greedy" Q-Learning to be used on various environments from the PettingZoo 
    library. This file was modified to support use of the sumo-rl traffic simulator library
    https://github.com/LucasAlegre/sumo-rl which is not technically part of the PettingZoo module but 
    conforms to the Petting Zoo API. Configuration of this script is performed through a configuration file, 
    examples of which can be found in the experiments/ directory.

    Note that experiments using the SUMO traffic simulator also require 'net' and 'route' files to configure 
    the environment.

Usage:
    python dqn-greedy.py -c experiments/sumo-2x2-dqn-greedy.config    

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
import gym
# TODO: fix conda environment to include the version of gym that has Monitor module
from gym.wrappers import TimeLimit#, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
from datetime import datetime
import random
import os
import csv
import pettingzoo

# SUMO dependencies
import sumo_rl
import sys

if __name__ == "__main__":
    parser = configargparse.ArgParser(default_config_files=['experiments/sumo-4x4-independent.config'], description='DQN agent')
    parser.add_argument('-c', '--config_path', required=False, is_config_file=True, help='config file path')

    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="CartPole-v0",
                        help='the id of the gym environment')
    parser.add_argument('--env-args', type=str, default="",
                        help='string to pass to env init')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=500000,
                        help='total timesteps of the experiments')
    parser.add_argument('--max-cycles', type=int, default=100,
                        help='max cycles in each step of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="DA-RL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--render', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, render environment')
    parser.add_argument('--global-obs', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, stack agent observations into global state')
    parser.add_argument('--gpu-id', type=str, default=None,
                        help='gpu device to use')
    parser.add_argument('--nn-save-freq', type=int, default=1000,
                        help='how often to save a copy of the neural network')

    # Algorithm specific arguments
    parser.add_argument('--N', type=int, default=3,
                        help='the number of agents')
    parser.add_argument('--buffer-size', type=int, default=10000,
                         help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=500,
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=32,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--start-e', type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument('--end-e', type=float, default=0.05,
                        help="the ending epsilon for exploration")
    parser.add_argument('--lam', type=float, default=0.01,
                        help="the pension for the variance")
    parser.add_argument('--exploration-fraction', type=float, default=0.05,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument('--learning-starts', type=int, default=10000,
                        help="timestep to start learning")
    parser.add_argument('--train-frequency', type=int, default=1,
                        help="the frequency of training")
    parser.add_argument('--load-weights', type=bool, default=False,
                    help="whether to load weights for the Q Network")

    # Configuration parameters specific to the SUMO traffic environment
    parser.add_argument("--route-file", dest="route", type=str, required=False, help="Route definition xml file.\n")
    parser.add_argument("--net-file", dest="net", type=str, required=False, help="Net definition xml file.\n")
    parser.add_argument("--mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum time for green lights in SUMO environment.\n")
    parser.add_argument("--maxgreen", dest="max_green", type=int, default=30, required=False, help="Maximum time for green lights in SUMO environment.\n")
    parser.add_argument("--sumo-gui", dest="sumo_gui", action="store_true", default=False, help="Run with visualization on SUMO (may require firewall permissions).\n")
    parser.add_argument("--sumo-seconds", dest="sumo_seconds", type=int, default=10000, required=False, help="Number of simulation seconds. The number of seconds the simulation must end.\n")
    parser.add_argument("--sumo-reward", dest="sumo_reward", type=str, default='wait', required=False, help="Reward function: \nThe 'queue'reward returns the negative number of total vehicles stopped at all agents each step, \nThe 'wait' reward returns the negative number of cummulative seconds that vehicles have been waiting in the episode.\n")

    # Configuration parameters for analyzing sumo env (only used in sumo_analysis.py)
    parser.add_argument("--analysis-steps", dest="analysis_steps", type=int, default=500, required=False, 
                        help="The number of time steps at which we want to investigate the perfomance of the algorithm. E.g. display how the training was going at the 10,000 checkpoint. Note there must be a nn .pt file for each agent at this step.\n")
    parser.add_argument("--nn-directory", dest="nn_directory", type=str, default=None, required=False, 
                        help="The directory containing the nn .pt files to load for analysis.\n")
    parser.add_argument("--parameter-sharing-model", dest="parameter_sharing_model", type=bool, default=False, required=True, 
                        help="Flag indicating if the model trained leveraged parameter sharing or not (needed to identify the size of the model to load).\n")

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
        args.seed = int(time.time())

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
    
# Instantiate the environment 
if using_sumo:
    # Sumo must be created using the sumo-rl module
    # Note we have to use the parallel env here to conform to this implementation of dqn
    # The 'queue' reward is being used here which returns the (negative) total number of vehicles stopped at all intersections
    env = sumo_rl.parallel_env(net_file=args.net, 
                    route_file=args.route,
                    use_gui=args.sumo_gui,
                    max_green=args.max_green,
                    min_green=args.min_green,
                    num_seconds=args.sumo_seconds,
                    reward_fn=args.sumo_reward, 
                    sumo_warnings=False)

else: 
    exec(f"import pettingzoo.{args.gym_id}") # lol
    exec(f"env = pettingzoo.{args.gym_id}.parallel_env({args.env_args})") # lol

agents = env.possible_agents
num_agents = len(env.possible_agents)
# TODO: these dictionaries are deprecated, use action_space & observation_space functions instead
action_spaces = env.action_spaces
observation_spaces = env.observation_spaces

print("\n=================== Environment Information ===================")
print(" > agents: {}".format(agents))
print(" > num_agents: {}".format(num_agents))
print(" > action_spaces: {}".format(action_spaces))
print(" > observation_spaces: {}".format(observation_spaces))

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
# env.action_space.seed(args.seed)
# env.observation_space.seed(args.seed)
for agent in agents:
    action_spaces[agent].seed(args.seed)
    observation_spaces[agent].seed(args.seed)
    #assert isinstance(action_spaces[agent], Discrete), "only discrete action space is supported"
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
        hidden_size = 64
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
optimizer = {}  # Dictionary for mapping agent to optimizer
neighbors = {agent: agents for agent in agents} # Dictionary that maps an agent to its "neighbors" TODO: should this be an experiment parameter?

# Initialize each data structure for each agent
for agent in agents:
    observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if args.global_obs else observation_spaces[agent].shape
    # print(observation_space_shape)
    rb[agent] = ReplayBuffer(args.buffer_size)
    q_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
    target_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
    target_network[agent].load_state_dict(q_network[agent].state_dict())
    optimizer[agent] = optim.Adam(q_network[agent].parameters(), lr=args.learning_rate)

loss_fn = nn.MSELoss()
print(device.__repr__())
print(q_network[agent]) # network of last agent

# TRY NOT TO MODIFY: start the game
obses = env.reset()

# Global states
if args.global_obs:
    global_obs = np.hstack(list(obses.values()))
    obses = {agent: global_obs for agent in agents}

# print(obses)
if args.render:
    env.render()

episode_rewards = {agent: 0 for agent in agents}
actions = {agent: None for agent in agents}
losses = {agent: None for agent in agents}
min_ind_rewards = 0
max_ind_rewards = 0
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
    next_obses, rewards, dones, _, _  = env.step(actions)

    # Global states
    if args.global_obs:
        global_obs = np.hstack(list(next_obses.values()))
        next_obses = {agent: global_obs for agent in agents}

    if args.render:
        env.render()

    # Extract performance about how we're doing so far
    min_ind_rewards += min(rewards.values())  # Accumulated min reward received by any agent this step
    max_ind_rewards += max(rewards.values())  # Accumulated max reward received by any agent this step
    
    for agent in agents:
        
        episode_rewards[agent] += rewards[agent]

        # ALGO LOGIC: training.
        rb[agent].put((obses[agent], actions[agent], rewards[agent], next_obses[agent], dones[agent]))
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            s_obses, s_actions, s_rewards, s_next_obses, s_dones = rb[agent].sample(args.batch_size)
            with torch.no_grad():
                target_maxes = []
                for neighbor in neighbors[agent]:
                    # NOTE: each call to 'forward' returns a list of target values (one for each action) for that neighbor
                    # target_maxes stores the maximum of the target values for each neighbor
                    # the reason this is called "greedy" Q learning is we want to use the max of all these maximums to
                    # update the Q-network of this agent
                    target_maxes.append(torch.max(target_network[neighbor].forward(s_next_obses), dim=1)[0])
                # "Greedy" means we take the max here
                target_max = torch.max(torch.stack(target_maxes, dim=0), dim=0)[0]
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
        system_episode_reward = sum(list(episode_rewards.values()))
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        print(f"global_step={global_step}, system_episode_reward={system_episode_reward}")
        diff_1 = max(episode_rewards.values())-min(episode_rewards.values())
        print(f"system_episode_diff_1={diff_1}")
        diff_2 = max_ind_rewards-min_ind_rewards
        print(f"system_episode_diff_2={diff_2}")
        print(f"max_ind_rewards={max_ind_rewards}")
        print(f"min_ind_rewards={min_ind_rewards}")
        max_ind_rewards = 0
        min_ind_rewards = 0

        # Logging should only be done after we've started training, up until then, the agents are just getting experience
        if global_step > args.learning_starts:
            for agent in agents:
                writer.add_scalar("charts/episode_reward/" + agent, episode_rewards[agent], global_step)
            writer.add_scalar("charts/epsilon/", epsilon, global_step)
            writer.add_scalar("charts/system_episode_reward/", system_episode_reward, global_step)

            with open(f"{csv_dir}/episode_reward.csv", "a", newline="") as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_reward', 'global_step'])
                csv_writer.writerow({**episode_rewards, **{'system_episode_reward': system_episode_reward, 'global_step': global_step}})

            # If we're using the SUMO env, also save some data specific to that environment
            if using_sumo:
                env.unwrapped.save_csv(sumo_csv, global_step)

        obses = env.reset()

        # Global states
        if args.global_obs:
            global_obs = np.hstack(list(obses.values()))
            obses = {agent: global_obs for agent in agents}

        if args.render:
            env.render()
        episode_rewards = {agent: 0 for agent in agents}
        actions = {agent: None for agent in agents}


env.close()
writer.close()

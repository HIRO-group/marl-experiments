"""
dqn-independent-ps-SUMO.py

Description:
    Implementation of independent Q-Learning with parameter sharing to be used on various environments from the PettingZoo 
    library. This file was modified from dqn-independent.py to support use of the sumo-rl traffic simulator library
    https://github.com/LucasAlegre/sumo-rl which is not technically part of the PettingZoo module but 
    conforms to the Petting Zoo API. Configuration of this script is performed through a configuration file, 
    examples of which can be found in the experiments/ directory.

    Note that experiments using the SUMO traffic simulator also require 'net' and 'route' files to configure 
    the environment.

Usage:
    python dqn-indepndent-ps-SUMO.py -c experiments/sumo-4x4-dqn-independent-ps.config    

References:
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
from gym.wrappers import TimeLimit #, Monitor
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
    parser.add_argument('--global-obs', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
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

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

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
    exec(f"env = pettingzoo.{args.gym_id}.parallel_env(N={args.N}, local_ratio=0.5, max_cycles={args.max_cycles}, continuous_actions=False)") # lol

agents = env.possible_agents
num_agents = len(env.possible_agents)
# TODO: these dictionaries are deprecated, use action_space & observation_space functions instead
action_spaces = env.action_spaces
observation_spaces = env.observation_spaces
onehot_keys = {agent: i for i, agent in enumerate(agents)}

print("\n=================== Environment Information ===================")
print("agents:\n {}".format(agents))
print("num_agents:\n {}".format(num_agents))
print("action_spaces:\n {}".format(action_spaces))
print("observation_spaces:\n {}".format(observation_spaces))

# TODO: Plotting loss is currently disabled for this implementation - see below
# with open(f"{csv_dir}/td_loss.csv", "w", newline="") as csvfile:
#     csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_loss', 'global_step'])
#     csv_writer.writeheader()
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
# respect the default timelimit
# assert isinstance(env.action_space, Discrete), "only discrete action space is supported"
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

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

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, observation_space_shape, action_space_dim):
        super(QNetwork, self).__init__()
        hidden_size = num_agents * 64 # TODO: should we make this a config parameter?
        self.fc1 = nn.Linear(np.array(observation_space_shape).prod(), hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, action_space_dim, bias=True)

    def forward(self, x):
        '''
        Propagate an observation through the Q network to produce a function that of an action
        '''
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

# neighbors = {agent: agents for agent in agents}

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
for agent in agents:
    rb[agent] = ReplayBuffer(args.buffer_size)
q_network = QNetwork(observation_space_shape, action_spaces[eg_agent].n).to(device) # In parameter sharing, all agents utilize the same q-network
target_network = QNetwork(observation_space_shape, action_spaces[eg_agent].n).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)   

loss_fn = nn.MSELoss()  # TODO: should the loss function be configurable?
print(device.__repr__())
print(q_network) # network of last agent

# TRY NOT TO MODIFY: start the game
obses = env.reset()


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
    epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)

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
        # The reward for the SUMO environment has been set to return the total (negative) number of cars waiting at each intersection 
        # So we don't want to accumulate it twice
        if using_sumo:
            episode_rewards[agent] = rewards[agent]
        else:
            episode_rewards[agent] += rewards[agent]

        rb[agent].put((obses[agent], actions[agent], rewards[agent], next_obses[agent], dones[agent]))
    
    # ALGO LOGIC: training.
    if global_step > args.learning_starts and global_step % args.train_frequency == 0:
        agent = random.choice(agents) # each minibatch is "centered" around a random agent
        # turn = int(global_step/num_turns)%num_agents    # Pick the agent around which the minibatch will be centered
        # agent = agents[turn]
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

        # TODO: plotting loss currently disabled for this implementation - see below
        # if global_step % 100 == 0:	
        #     writer.add_scalar("losses/td_loss/" + agent, loss, global_step)

        #     with open(f"{csv_dir}/td_loss.csv", "a", newline="") as csvfile:
        #         csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['global_step'])
        #         csv_writer.writerow({agent: loss.item(), 'global_step': global_step})

        # optimize the midel
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

    # TODO: Plossting td_loss is currently disabled for this implementation - it must be updated to suppor tracking the total loss 
    # of the system rather than retrieving loss for a given agent
    # if global_step > args.learning_starts and global_step % args.train_frequency == 0:
    #     if global_step % 100 == 0:
    #         system_loss = sum(list(losses.values()))
    #         writer.add_scalar("losses/system_td_loss/", system_loss, global_step)

    #         with open(f"{csv_dir}/td_loss.csv", "a", newline="") as csvfile:
    #             csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_loss', 'global_step'])
    #             csv_writer.writerow({**losses, **{'system_loss': system_loss, 'global_step': global_step}})

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
        lir_1 = 0
        uir_1 = 0
        var_1 = 0
        cnt = 0

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
            env.render()
        episode_rewards = {agent: 0 for agent in agents}
        actions = {agent: None for agent in agents}


env.close()
writer.close()

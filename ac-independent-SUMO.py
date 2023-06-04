"""
ac-independent-SUMO.py

Description:
    Implementation of soft actor critic adapted for multi-agent environments. This implementation is origianlly based on
    the Clean-RL version https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py

Usage:


References:
    - https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py 

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical


import configargparse
from distutils.util import strtobool
import collections
import numpy as np

# # TODO: fix conda environment to include the version of gym that has Monitor module
# from gym.wrappers import TimeLimit#, Monitor
# from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
from datetime import datetime
import random
import os
import csv
# import pettingzoo
from pettingzoo.butterfly import pistonball_v6
from pettingzoo.mpe import simple_spread_v3

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
os.makedirs(f"{nn_dir}/critic_networks")
os.makedirs(f"{nn_dir}/actor_networks")
os.makedirs(csv_dir)

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
# device = torch.device('cpu')
print("Device: ", device)

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
    # exec(f"from pettingzoo.mpe import {args.gym_id}") # lol
    # exec(f"env = {args.gym_id}.parallel_env(N={args.N}, local_ratio=0.5, max_cycles={args.max_cycles}, continuous_actions=False)") # lol
    print(" > ENV ARGS: {}".format(args.env_args))
    exec(f"env = {args.gym_id}.parallel_env({args.env_args})")

# from pettingzoo.mpe import simple_spread_v3
# from pettingzoo.butterfly import pistonball_v6
# print("ENV IMPORTED")

# env = simple_spread_v3.parallel_env(N=args.N, local_ratio=0.5, max_cycles=args.max_cycles, continuous_actions=False, render_mode=None)
# env = pistonball_v6.parallel_env(n_pistons=args.N, time_penalty=-0.1, continuous=False, max_cycles=args.max_cycles, render_mode=None)
# print("ENV CREATED")

print("\n=================== Environment Information ===================")
agents = env.possible_agents
print(" > agents:\n {}".format(agents))

num_agents = len(env.possible_agents)
print(" > num_agents:\n {}".format(num_agents))

# TODO: these dictionaries are deprecated, use action_space & observation_space functions instead
action_spaces = env.action_spaces
print(" > action_spaces:\n {}".format(action_spaces))

observation_spaces = env.observation_spaces
print(" > observation_spaces:\n {}".format(observation_spaces))


with open(f"{csv_dir}/critic_loss.csv", "w", newline="") as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_loss', 'global_step'])
    csv_writer.writeheader()
with open(f"{csv_dir}/actor_loss.csv", "w", newline="") as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_actor_loss', 'global_step'])
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
# This is the Critic
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

# Define the Actor class
# Based on implementation from here: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py
class Actor(nn.Module):
    def __init__(self, observation_space_shape, action_space_dim):
        super(Actor, self).__init__()
        hidden_size = 64
        self.fc1 = nn.Linear(np.array(observation_space_shape).prod(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_logits = nn.Linear(hidden_size, action_space_dim)

    def forward(self, x):
        # x = torcsh.Tensor(x).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # print(">> SHAPE OF X: {}".format(x.shape))
        logits = self.fc_logits(x)
        # print(">> SHAPE OF LOGITS: {}".format(logits.shape))
        return logits
    
    def get_action(self, x):
        x = torch.Tensor(x).to(device)
        logits = self.forward(x)
        # Note that this is equivalent to what used to be called multinomial (i.e. what softmax produces)
        policy_dist = Categorical(logits=logits)
        # print(" >>> Categorical: {}".format(policy_dist))
        # print(" >>> softmax: {}".format(F.softmax(logits)))
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=-1)
        # return action, torch.transpose(log_prob, 0, 1), action_probs
        return action, log_prob, action_probs



def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    '''
    Defines a schedule for decaying epsilon during the training procedure
    '''
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

# Initialize data structures for training
rb = {} # Dictionary for storing replay buffers (maps agent to a replay buffer)
q_network = {}  # Dictionary for storing q-networks (maps agent to a q-network), these are the "critics"
target_network = {} # Dictionary for storing target networks (maps agent to a network)
actor_network = {} # Dictionary for storing actor networks (maps agents to a network)
optimizer = {}  # Dictionary for storing optimizer for each agent's network
actor_optimizer = {} # Dictionary for storing the optimizers used to train the actor networks 

print(" > INITIALIZING NEURAL NETWORKS")
for agent in agents:
    print(" >> AGENT: {}".format(agent))
    observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if args.global_obs else observation_spaces[agent].shape
    print(" >> OBSERVATION_SPACE_SHAPE: {}".format(observation_space_shape))
    print(" >> ACTION_SPACE_SHAPE: {}".format(action_spaces[agent].n))
    rb[agent] = ReplayBuffer(args.buffer_size)
    q_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
    target_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
    target_network[agent].load_state_dict(q_network[agent].state_dict())    # Intialize the target network the same as the critic network
    actor_network[agent] = Actor(observation_space_shape, action_spaces[agent].n).to(device)
    # actor_network[agent].load_state_dict(q_network[agent].state_dict())   # Initialilze the actor network the same as the critic network
    optimizer[agent] = optim.Adam(q_network[agent].parameters(), lr=args.learning_rate) # All agents use the same optimizer for training
    actor_optimizer[agent] = optim.Adam(list(actor_network[agent].parameters()), lr=args.learning_rate)

loss_fn = nn.MSELoss() # TODO: should the loss function be configurable?
actor_loss_fn = nn.CrossEntropyLoss()

print(" > Device: ",device.__repr__())
print(" > Q_network structure: ", q_network[agent]) # network of last agent

# TRY NOT TO MODIFY: start the game
obses, _ = env.reset()
# obses = env.reset()

# print(" >> RAW OBSERVATION: {}".format(obses))
# print(" >> len(obses): {}".format(len(obses)))
# print(" >> obses[0]: {}".format(obses[0]))
# print(" >> obses[1]: {}".format(obses[1]))

# obses = obses[0]    #TODO: For some reason obses is a tuple here for the pistonball env
# for agent in agents:
#     print(" >> SHAPES AFTER: {}".format(obses[agent].shape))


# Global states
if args.global_obs:
    print(" >> NOTE: USING GLOBAL OBSERVATIONS ")
    global_obs = np.hstack(list(obses.values()))
    obses = {agent: global_obs for agent in agents}

else: 
    print(" >> GLOBAL OBSERVATIONS DISABLED")

if args.render:
    env.render()    # TODO: verify that the sumo env supports render

episode_rewards = {agent: 0 for agent in agents}    # Dictionary that maps the each agent to its cumulative reward each episode
actions = {agent: None for agent in agents}
losses = {agent: None for agent in agents}          # Dictionary that maps each agent to the loss values for its critic network
actor_losses = {agent: None for agent in agents}    # Dictionary that maps each agent to the loss values for its actor network
lir_1 = 0
uir_1 = 0
var_1 = 0
cnt = 0

# TODO: this is the "entropy regularization coefficient", make this configurable or add the "autotune" from the CleanRL version
entropy_reg_coef = 0.2 

for global_step in range(args.total_timesteps):

    # ALGO LOGIC: put action logic here
    epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)

    # Set the action for each agent
    for agent in agents:
        if random.random() < epsilon:
            actions[agent] = action_spaces[agent].sample()
        else:
            action, _, _ = actor_network[agent].get_action(obses[agent])
            actions[agent] = action.detach().cpu().numpy()
            # logits = q_network[agent].forward(obses[agent].reshape((1,)+obses[agent].shape))  # Used in SUMO but not in simple_spread
            # actions[agent] = torch.argmax(logits, dim=1).tolist()[0]

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obses, rewards, dones, _, _ = env.step(actions)

    # print(" >>> obses[{}]: {}".format(agent,obses[agent]))
    # print(" >>> actions[{}]: {} ".format(agent,actions[agent]))
    # print(" >>> next_obses[{}]: {} ".format(agent,next_obses[agent]))
    # print(" >>> rewards[{}]: {} ".format(agent,rewards[agent]))
    # print(" >>> dones[{}]: {} ".format(agent,dones[agent]))

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
        # The reward for the SUMO environment has been set to return the total (negative) number of cars waiting at each intersection 
        # So we don't want to accumulate it twice
        if using_sumo:
            episode_rewards[agent] = rewards[agent]
        else:
            episode_rewards[agent] += rewards[agent]

        # ALGO LOGIC: critic training
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

            # optimize the model for the critic
            optimizer[agent].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(q_network[agent].parameters()), args.max_grad_norm)
            optimizer[agent].step()


            # Actor training
            a, log_pi, action_probs = actor_network[agent].get_action(s_obses)
            # TODO: how should these be calculated?
            # Using critic network? Target network?
            # S_obses or obses?
            with torch.no_grad():
                values = q_network[agent].forward(obses[agent])
                # values = q_network[agent].forward(s_obses)

            actor_loss = (action_probs * ((entropy_reg_coef * log_pi) - values)).mean() # Modified from CleanRL SAC

            # Actor uses cross-entropy loss function where
            # input is the policy dist and the target is the value function with one-hot encoding applied
            # actor_loss = actor_loss_fn(action_probs, F.one_hot(values))
            # print(" >>> Values: {}".format(values))
            # print(" >>> Values ONE HOT: {}".format(values))
            # print( " >> actor_loss = (action_probs * (entropy_reg_coef * log_pi) - values).mean() = {}".format((actor_loss)))
            actor_losses[agent] = actor_loss.item()

            actor_optimizer[agent].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(list(actor_network[agent].parameters()), args.max_grad_norm)
            actor_optimizer[agent].step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network[agent].load_state_dict(q_network[agent].state_dict())

        # Save a snapshot of the actor and critic networks at this iteration of training
        if global_step % args.nn_save_freq == 0:
            for a in agents:
                torch.save(q_network[a].state_dict(), f"{nn_dir}/critic_networks/{global_step}-{a}.pt")
                torch.save(actor_network[a].state_dict(), f"{nn_dir}/actor_networks/{global_step}-{a}.pt")

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook 
    obses = next_obses

    if global_step > args.learning_starts and global_step % args.train_frequency == 0:
        if global_step % 100 == 0:
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

    # If all agents are done, log the results and reset the evnironment to continue training
    if np.prod(list(dones.values())) or global_step % args.max_cycles == args.max_cycles-1: 
        system_episode_reward = sum(list(episode_rewards.values())) # Accumulated reward of all agents

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        print(f"global_step={global_step}, system_episode_reward={system_episode_reward}")
        diff_1 = uir_1-lir_1
        # var_1 = var_1/(cnt-1e-7)
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
        actions = {agent: None for agent in agents}


env.close()
writer.close()

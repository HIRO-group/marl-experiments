"""
TODO: Either delete this file, update it to conform to the rest of the repository, or move it to an 
"under_construction" directory or something
"""

# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import collections
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
import csv
import pettingzoo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
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
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

def one_hot(a, size):
    b = np.zeros((size))
    b[a] = 1
    return b

# class ProcessObsInputEnv(gym.ObservationWrapper):
#     """
#     This wrapper handles inputs from `Discrete` and `Box` observation space.
#     If the `env.observation_space` is of `Discrete` type, 
#     it returns the one-hot encoding of the state
#     """
#     def __init__(self, env):
#         super().__init__(env)
#         self.n = None
#         if isinstance(self.env.observation_space, Discrete):
#             self.n = self.env.observation_space.n
#             self.observation_space = Box(0, 1, (self.n,))

#     def observation(self, obs):
#         if self.n:
#             return one_hot(np.array(obs), self.n)
#         return obs

# TRY NOT TO MODIFY: setup the environment
if args.gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
experiment_name = f"{args.gym_id}__N={args.N}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

# env = ProcessObsInputEnv(gym.make(args.gym_id))
exec(f"import pettingzoo.{args.gym_id}") # lol
exec(f"env = pettingzoo.{args.gym_id}.parallel_env(N={args.N}, local_ratio=0.5, max_cycles={args.max_cycles}, continuous_actions=False)") # lol

agents = env.possible_agents
num_agents = len(env.possible_agents)
action_spaces = env.action_spaces
observation_spaces = env.observation_spaces

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
    # assert isinstance(action_spaces[agent], Discrete), "only discrete action space is supported"
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

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, observation_space_shape, action_space_dim):
        super(QNetwork, self).__init__()
        # hidden_size = num_agents * 64
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
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


rb = {}
q_network = {}
target_network = {}
optimizer = {}
neighbors = {agent: agents for agent in agents}

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
lir_1 = 0
uir_1 = 0
var_1 = 0
cnt = 0
num_turns = 57
for global_step in range(args.total_timesteps):

    # ALGO LOGIC: put action logic here
    epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)

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
    lir_1 += min(rewards.values())
    uir_1 += max(rewards.values())    
    var_1 += np.var(list(rewards.values()))
    cnt += 1
    for agent in agents:
        episode_rewards[agent] += rewards[agent]
    # ALGO LOGIC: training.
    turn = int(global_step/num_turns)%num_agents
    mAagent = agents[turn]
    
    rb[mAagent].put((obses[mAagent], actions[mAagent], rewards[mAagent], next_obses[mAagent], dones[mAagent]))
    
    if global_step > args.learning_starts and global_step % args.train_frequency == 0:
        s_obses, s_actions, s_rewards, s_next_obses, s_dones = rb[mAagent].sample(args.batch_size)
        with torch.no_grad():    
            target = torch.max(target_network[mAagent].forward(s_next_obses), dim=1)[0]
            pen = target-(-100)
            target = torch.where(pen<0, target - args.lam*pen, target)
            td_target = torch.Tensor(s_rewards).to(device) + args.gamma * target * (1 - torch.Tensor(s_dones).to(device))
        old_val = q_network[mAagent].forward(s_obses).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
        loss = loss_fn(td_target, old_val)
        losses[mAagent] = loss.item()

        if global_step % 100 == 0:
            writer.add_scalar("losses/td_loss/" + mAagent, loss, global_step)

        # optimize the midel
        optimizer[mAagent].zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(q_network[mAagent].parameters()), args.max_grad_norm)
        optimizer[mAagent].step()

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

    if np.prod(list(dones.values())) or global_step % args.max_cycles == args.max_cycles-1: # all agents done
        system_episode_reward = sum(list(episode_rewards.values()))

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
        lir_1 = 0
        uir_1 = 0
        var_1 = 0
        cnt = 0
        with open(f"{csv_dir}/episode_reward.csv", "a", newline="") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_reward', 'global_step'])
            csv_writer.writerow({**episode_rewards, **{'system_episode_reward': system_episode_reward, 'global_step': global_step}})

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

# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
import csv
import pettingzoo

from replay_buffer import ReplayBuffer
from linear_schedule import LinearSchedule

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
    
# respect the default timelimit
# assert isinstance(env.action_space, Discrete), "only discrete action space is supported"
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')


# TODO: THis file needs to be updated more before we can replace this QNetwork class with the one 
# in actor_critic.py (notice the size of the hidden layer)
class QNetwork(nn.Module):
    def __init__(self, observation_space_shape, action_space_dim):
        super(QNetwork, self).__init__()
        hidden_size = num_agents * 64
        self.fc1 = nn.Linear(np.array(observation_space_shape).prod(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space_dim)

    def forward(self, x):
        x = torch.Tensor(x).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if args.global_obs else observation_spaces[agent].shape
action_space_shape = np.prod([space.n for space in action_spaces.values()])
rb = ReplayBuffer(args.buffer_size)
q_network = QNetwork(observation_space_shape, action_space_shape).to(device)
target_network = QNetwork(observation_space_shape, action_space_shape).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

loss_fn = nn.MSELoss()
print(device.__repr__())
print(q_network) # network of last agent

# TRY NOT TO MODIFY: start the game
obses = env.reset()

# Global states
if args.global_obs:
    global_obs = np.hstack(list(obses.values()))

if args.render:
    env.render()
episode_rewards = {agent: 0 for agent in agents}
actions = {agent: None for agent in agents}

lir_1 = 0
uir_1 = 0
var_1 = 0
cnt = 0
for global_step in range(args.total_timesteps):

    # ALGO LOGIC: put action logic here
    epsilon = LinearSchedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)

    if random.random() < epsilon:
        action = np.random.randint(action_space_shape)
    else:
        logits = q_network.forward(global_obs.reshape((1,)+global_obs.shape))
        action = torch.argmax(logits, dim=1).tolist()[0]

    # TRY NOT TO MODIFY: execute the game and log data.
    tmp_action = action
    divisor = action_spaces[agents[0]].n # Hack! Won't work if action spaces different size
    for agent in agents:
        actions[agent] = tmp_action % divisor
        tmp_action = tmp_action // divisor
    next_obses, rewards, dones, _, _ = env.step(actions)
    central_reward = sum(list(rewards.values()))
    done = np.prod(list(dones.values()))

    # Global states
    if args.global_obs:
        global_next_obs = np.hstack(list(next_obses.values()))

    if args.render:
        env.render()

    lir_1 += min(rewards.values())
    uir_1 += max(rewards.values()) 
    var_1 += np.var(list(rewards.values()))
    cnt += 1
    for agent in agents:
        episode_rewards[agent] += rewards[agent]

    # ALGO LOGIC: training.
    rb.put((global_obs, action, central_reward, global_next_obs, done))
    if global_step > args.learning_starts and global_step % args.train_frequency == 0:
        s_obses, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
        with torch.no_grad():
            target_max = torch.max(target_network.forward(s_next_obses), dim=1)[0]
            td_target = torch.Tensor(s_rewards).to(device) + args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))
        old_val = q_network.forward(s_obses).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
        loss = loss_fn(td_target, old_val)

        if global_step % 100 == 0:
            writer.add_scalar("losses/system_td_loss/", loss, global_step)

            with open(f"{csv_dir}/td_loss.csv", "a", newline="") as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=['system_loss', 'global_step'])
                csv_writer.writerow({'system_loss': loss.item(), 'global_step': global_step})

        # optimize the midel
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
        optimizer.step()

        # update the target network
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    if global_step % args.nn_save_freq == 0:
        torch.save(q_network.state_dict(), f"{nn_dir}/{global_step}.pt")

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook 
    global_obs = global_next_obs

    if done or global_step % args.max_cycles == args.max_cycles-1: # all agents done
        system_episode_reward = sum(list(episode_rewards.values()))

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
        for agent in agents:
            writer.add_scalar("charts/episode_reward/" + agent, episode_rewards[agent], global_step)
        writer.add_scalar("charts/epsilon/", epsilon, global_step)
        writer.add_scalar("charts/system_episode_reward/", system_episode_reward, global_step)

        with open(f"{csv_dir}/episode_reward.csv", "a", newline="") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_reward', 'global_step'])
            csv_writer.writerow({**episode_rewards, **{'system_episode_reward': system_episode_reward, 'global_step': global_step}})

        obses = env.reset()

        # Global states
        if args.global_obs:
            global_obs = np.hstack(list(obses.values()))

        if args.render:
            env.render()
        episode_rewards = {agent: 0 for agent in agents}
        actions = {agent: None for agent in agents}


env.close()
writer.close()

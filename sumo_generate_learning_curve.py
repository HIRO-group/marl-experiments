"""
sumo_generate_learning_curve.py

Description:
    File for generating the learning curve for a multi-agent model trained on the SUMO environment.
    This script evaluates the neural networks that were periodically saved during the training process
    by executing each checkpoint on the SUMO environment. The SUMO environment is configured the same way
    here as it is in the training process but here agents are not allowed to take epsilon-random actions 
    (actions are chosen soley from the model's policy). 

Usage:
    python sumo_generate_learning_curve.py -c experiments/sumo-2x2-ac-independent.config    
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import configargparse
from distutils.util import strtobool
import numpy as np
from datetime import datetime
from torch.distributions.categorical import Categorical

import random
import os
import csv
import pettingzoo

# SUMO dependencies
import sumo_rl
import sys
from sumo_custom_observation import CustomObservationFunction
from sumo_custom_reward import MaxSpeedRewardFunction


# Make sure SUMO env variable is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


# # TODO: this should probably just go in its own file so it's consistent across all training
# # TODO: May need to update this for actor critic, actor and critic should have the same "forward" structure
# class QNetwork(nn.Module):
#     def __init__(self, observation_space_shape, action_space_dim, parameter_sharing_model=False):
#         super(QNetwork, self).__init__()
#         self.parameter_sharing_model = parameter_sharing_model
#         # Size of model depends on if parameter sharing was used or not
#         if self.parameter_sharing_model:
#             hidden_size = num_agents * 64
#         else:
#             hidden_size = 64    # TODO: should we make this a config parameter?
#         self.fc1 = nn.Linear(np.array(observation_space_shape).prod(), hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, action_space_dim)

#     def forward(self, x):
#         x = torch.Tensor(x).to(device)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# TODO: add config flag to indicate if the Actor model or the critic model should be used
# Define the Actor class
# Based on implementation from here: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py
class Actor(nn.Module):
    def __init__(self, observation_space_shape, action_space_dim):
        super(Actor, self).__init__()
        hidden_size = 64
        self.fc1 = nn.Linear(np.array(observation_space_shape).prod(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # print(">> SHAPE OF X: {}".format(x.shape))
        logits = self.fc3(x)
        # print(">> SHAPE OF LOGITS: {}".format(logits.shape))
        return logits
    
    def get_action(self, x):
        x = torch.Tensor(x).to(device)
        logits = self.forward(x)
        # Note that this is equivalent to what used to be called multinomial 
        # policy_dist.probs here will produce the same thing as softmax(logits)
        policy_dist = Categorical(logits=logits)
        # policy_dist = F.softmax(logits)
        # print(" >>> Categorical: {}".format(policy_dist.probs))
        # print(" >>> softmax: {}".format(F.softmax(logits)))
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=-1)
        # return action, torch.transpose(log_prob, 0, 1), action_probs
        return action, log_prob, action_probs


if __name__ == "__main__":
    parser = configargparse.ArgParser(default_config_files=['experiments/sumo-4x4-independent.config'], 
                                      description="Generate the learning curve for agents trained on the SUMO environment")
    parser.add_argument('-c', '--config_path', required=False, is_config_file=True, help='config file path')

     # TODO: remove unecessary configs here, we're just looking at sumo in this file
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

    # Configuration parameters for analyzing sumo env
    parser.add_argument("--analysis-steps", dest="analysis_steps", type=int, default=500, required=True, 
                        help="The number of time steps at which we want to investigate the perfomance of the algorithm. E.g. display how the training was going at the 10,000 checkpoint. Note there must be a nn .pt file for each agent at this step.\n")
    parser.add_argument("--nn-directory", dest="nn_directory", type=str, default=None, required=True, 
                        help="The directory containing the nn .pt files to load for analysis.\n")
    parser.add_argument("--parameter-sharing-model", dest="parameter_sharing_model", action="store_true", default=False, required=False, 
                        help="Flag indicating if the model trained leveraged parameter sharing or not (needed to identify the size of the model to load).\n")
                        

    args = parser.parse_args()
    
    # Create CSV file to store the data
    nn_directory = args.nn_directory        
    nn_dir = f"nn/{nn_directory}"           # Name of directory where the nn was stored during training    
    analysis_time = str(datetime.now()).split('.')[0].replace(':','-')
    csv_dir = f"analysis/{nn_directory+analysis_time}"
    os.makedirs(csv_dir)

    if not args.seed:
        args.seed = int(datetime.now()) 
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    parameter_sharing_model = args.parameter_sharing_model  # Flag indicating if we're loading a model from DQN with PS
    print(" > Parameter Sharing Enabled: {}".format(parameter_sharing_model))

    # Create the env
    # Sumo must be created using the sumo-rl module
    # Note we have to use the parallel env here to conform to this implementation of dqn
    # The 'queue' reward is being used here which returns the (negative) total number of vehicles stopped at all intersections
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

    # Seed the env
    env.reset(seed=args.seed)
    for agent in agents:
        action_spaces[agent].seed(args.seed)
        observation_spaces[agent].seed(args.seed)

    # Initialize the CSV header for the learning curve
    with open(f"{csv_dir}/learning_curve.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_reward', 'nn_step'])
        csv_writer.writeheader()

    # Initialize the csv header for the max speeds file
    with open(f"{csv_dir}/episode_max_speeds.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_max_speed', 'system_episode_min_max_speed', 'nn_step'])    
        csv_writer.writeheader()

    # Loop over all the nn files in the nn directory
    # Every nn_save_freq steps, a new nn.pt file was saved during training
    for saved_step in range(0, args.total_timesteps, args.nn_save_freq):
        print(" > Loading network at learning step: {}".format(saved_step))

        onehot_keys = {agent: i for i, agent in enumerate(agents)}
        episode_rewards = {agent: 0 for agent in agents}            # Dictionary that maps the each agent to its cumulative reward each episode
        episode_max_speeds = {agent: [] for agent in agents}        # Dictionary that maps each agent to the maximum speed observed at each step of the agent's episode

        # Construct the Q-Network model 
        # Note the dimensions of the model varies depending on if the parameter sharing algorithm was used or the normal independent 
        # DQN model was used
        if parameter_sharing_model:
            # Define the shape of the observation space depending on if we're using a global observation or not
            # Regardless, we need to add an array of length num_agents to the observation to account for one hot encoding
            eg_agent = agents[0]
            if args.global_obs:
                observation_space_shape = tuple((shape+1) * (num_agents) for shape in observation_spaces[eg_agent].shape)
            else:
                observation_space_shape = np.array(observation_spaces[eg_agent].shape).prod() + num_agents  # Convert (X,) shape from tuple to int so it can be modified
                observation_space_shape = tuple(np.array([observation_space_shape]))                        # Convert int to array and then to a tuple
    
            # q_network = QNetwork(observation_space_shape, action_spaces[eg_agent].n, parameter_sharing_model).to(device) # In parameter sharing, all agents utilize the same q-network
            q_network = Actor(observation_space_shape, action_spaces[eg_agent].n, parameter_sharing_model).to(device) # In parameter sharing, all agents utilize the same q-network

            # Load the Q-network file
            nn_file = "{}/{}.pt".format(nn_dir, saved_step)
            q_network.load_state_dict(torch.load(nn_file))

        # Else the agents were trained using normal independent DQN so each agent gets its own Q-network model
        else: 
            
            q_network = {}  # Dictionary for storing q-networks (maps agent to a q-network)
            
            # Load the Q-Network NN model for each agent from the specified anaylisis checkpoint step from training
            for agent in agents: 
                observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if args.global_obs else observation_spaces[agent].shape
                # q_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n)
                q_network[agent] = Actor(observation_space_shape, action_spaces[agent].n).to(device) # In parameter sharing, all agents utilize the same q-network

                nn_file = "{}/{}-{}.pt".format(nn_dir, saved_step, agent)   # TODO: make the step number configurable
                q_network[agent].load_state_dict(torch.load(nn_file))

        # Initialize the env
        print(" > Resetting environment")
        obses, _ = env.reset()

        # Initialize observations depending on if parameter sharing was used or not
        if parameter_sharing_model:
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
        
        # Parameter sharing model not used but we still need to check if global observations were used
        else:
            if args.global_obs:
                global_obs = np.hstack(list(obses.values()))
                obses = {agent: global_obs for agent in agents}

        # Define empty dictionary tha maps agents to actions
        actions = {agent: None for agent in agents}

        # Simulate the environment using actions derived from the Q-Network
        print(" > Executing policy from network")
        for sumo_step in range(args.sumo_seconds):
            # Populate the action dictionary
            for agent in agents:
                # TODO: uncomment once the critic/actor flag is added
                # # # if parameter_sharing_model:
                # # #     logits = q_network.forward(obses[agent].reshape((1,)+obses[agent].shape))
                # # # else:
                # # #     logits = q_network[agent].forward(obses[agent].reshape((1,)+obses[agent].shape))

                # # # actions[agent] = torch.argmax(logits, dim=1).tolist()[0]

                ## Testing actor version
                action, _, _ = q_network[agent].get_action(obses[agent])
                actions[agent] = action.detach().cpu().numpy()


            # Apply all actions to the env
            next_obses, rewards, dones, _, _ = env.step(actions)

            # If the parameter sharing model was used, we have to add one hot encoding to the observations
            if parameter_sharing_model:
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
    
            # Accumulate the total episode reward and max speeds
            for agent in agents:
                
               
               # At the end of a simulation, next_obses is an empty dictionary so don't log it
                try:
                    episode_rewards[agent] += rewards[agent]
                    
                     # TODO: need to modify this for global observations
                    episode_max_speeds[agent].append(next_obses[agent][-1]) # max speed is the last element of the custom observation array
                
                except:
                    continue
                
                # print(" >>> episode_rewards: {}".format(episode_rewards))
                # print(" >>> rewards[{}]: {}".format(agent, rewards[agent]))

            obses = next_obses

            # If the simulation is done, print the episode reward and close the env
            if np.prod(list(dones.values())):
                print(" > Episode complete - logging data")

                system_episode_reward = sum(list(episode_rewards.values())) # Accumulated reward of all agents
                
                # Calculate the maximum of all max speeds observed from each agent during the episode
                agent_max_speeds = {agent:0 for agent in agents}
                for agent in agents:
                    agent_max_speeds[agent] = max(episode_max_speeds[agent])

                system_episode_max_speed = max(list(agent_max_speeds.values()))
                system_episode_min_max_speed = min(list(agent_max_speeds.values()))

                # Log the episode reward to CSV
                with open(f"{csv_dir}/learning_curve.csv", "a", newline="") as csvfile:
                    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_reward', 'nn_step'])
                    csv_writer.writerow({**episode_rewards, **{'system_episode_reward': system_episode_reward, 'nn_step': saved_step}})
                
                # Log the max speeds
                with open(f"{csv_dir}/episode_max_speeds.csv", "a", newline="") as csvfile:
                    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_max_speed', 'system_episode_min_max_speed', 'nn_step'])
                    csv_writer.writerow({**agent_max_speeds, **{'system_episode_max_speed': system_episode_max_speed,
                                                            'system_episode_min_max_speed': system_episode_min_max_speed,
                                                            'nn_step': saved_step}})

                print(" >> TOTAL EPISODE REWARD: {}\n".format(system_episode_reward))

                break
    
    env.close()
"""
sumo_analysis.py

Description:
    Script for analyzing the progress of MARL algorithms on the SUMO environment. This script is intended to load a NN model
    from a specified training checkpoint and display its performance in the SUMO GUI.

Usage:
    python sumo_analysis.py -c experiments/sumo-4x4-dqn-independent.config

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import datetime

import os

# SUMO dependencies
import sumo_rl
import sys

from sumo_custom_observation import CustomObservationFunction
from sumo_custom_reward import MaxSpeedRewardFunction

# Config Parser
from MARLConfigParser import MARLConfigParser

# Make sure SUMO env variable is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


# TODO: this should probably just go in its own file so it's consistent across all training
# TODO: May need to update this for actor critic, actor and critic should have the same "forward" structure
class QNetwork(nn.Module):
    def __init__(self, observation_space_shape, action_space_dim, parameter_sharing_model=False):
        super(QNetwork, self).__init__()
        self.parameter_sharing_model = parameter_sharing_model
        # Size of model depends on if parameter sharing was used or not
        if self.parameter_sharing_model:
            hidden_size = num_agents * 64
        else:
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


if __name__ == "__main__":
    
    # Get config parameters                        
    parser = MARLConfigParser()
    args = parser.parse_args()
    
    if not args.seed:
        args.seed = int(datetime.now()) 
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')


    analysis_steps = args.analysis_steps    # Defines which checkpoint will be loaded into the Q model
    parameter_sharing_model = args.parameter_sharing_model  # Flag indicating if we're loading a model from DQN with PS
    nn_directory = args.nn_directory 
    nn_dir = f"nn/{nn_directory}" # Name of directory where the nn was stored during training    

    print(">> PS? {}".format(parameter_sharing_model))

    # Create the env
    # Sumo must be created using the sumo-rl module
    # Note we have to use the parallel env here to conform to this implementation of dqn
    # The 'queue' reward is being used here which returns the (negative) total number of vehicles stopped at all intersections
    if (args.sumo_reward == "custom"):
        # Use the custom "max speed" reward function
        print ( " > Evaluating model using CUSTOM reward")
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
        print ( " > Evaluating model using standard reward: {}".format(args.sumo_reward))
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
    onehot_keys = {agent: i for i, agent in enumerate(agents)}
    
    episode_rewards = {agent: 0 for agent in agents}    # Dictionary that maps the each agent to its cumulative reward each episode

    print("\n=================== Environment Information ===================")
    print(" > agents:\n {}".format(agents))
    print(" > num_agents:\n {}".format(num_agents))
    print(" > action_spaces:\n {}".format(action_spaces))
    print(" > observation_spaces:\n {}".format(observation_spaces))

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
 
        q_network = QNetwork(observation_space_shape, action_spaces[eg_agent].n, parameter_sharing_model).to(device) # In parameter sharing, all agents utilize the same q-network
        
        # Load the Q-network file
        nn_file = "{}/{}.pt".format(nn_dir, analysis_steps)
        q_network.load_state_dict(torch.load(nn_file))

    # Else the agents were trained using normal independent DQN so each agent gets its own Q-network model
    else: 
        
        q_network = {}  # Dictionary for storing q-networks (maps agent to a q-network)
        
        # Load the Q-Network NN model for each agent from the specified anaylisis checkpoint step from training
        for agent in agents: 
            observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if args.global_obs else observation_spaces[agent].shape
            q_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n)

            nn_file = "{}/{}-{}.pt".format(nn_dir, analysis_steps, agent)   # TODO: make the step number configurable
            q_network[agent].load_state_dict(torch.load(nn_file))
    

    # Seed the env
    env.reset(seed=args.seed)
    for agent in agents:
        action_spaces[agent].seed(args.seed)
        observation_spaces[agent].seed(args.seed)

    # Initialize the env
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
    for sumo_step in range(args.sumo_seconds):
        # Populate the action dictionary
        for agent in agents:
            if parameter_sharing_model:
                logits = q_network.forward(obses[agent].reshape((1,)+obses[agent].shape))
            else:
                logits = q_network[agent].forward(obses[agent].reshape((1,)+obses[agent].shape))

            actions[agent] = torch.argmax(logits, dim=1).tolist()[0]

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
        
        # Accumulate the total episode reward
        for agent in agents:
            episode_rewards[agent] += rewards[agent]

        obses = next_obses

        # If the simulation is done, print the episode reward and close the env
        if np.prod(list(dones.values())):
            system_episode_reward = sum(list(episode_rewards.values())) # Accumulated reward of all agents

            print(" >> TOTAL EPISODE REWARD: {}".format(system_episode_reward))

            break
    
    env.close()

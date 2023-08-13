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

# Config Parser
from MARLConfigParser import MARLConfigParser

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


def CalculateASOMax(episode_max_speeds, speed_limit):
    """
    Function for calculating The average maximum speed overage (ASOmax) after an episode,
    ASO max is essentially the average amount that each agent execeeded a given speed limit over the course of an entire episode.
    This metric is used in part to evaulate the performance of models that were trained using the "custom speed threshold" reward defined 
    for the SUMO enviornment

    Note that the speed limit here does not need to match the speed threshold used to train the model
    
    :param episode_max_speeds: Dictionary that maps agents to the max speeds of cars observed each step of an episode
    :param speed_limit: The speed limit to use for comparison in the calculation of ASO max
    :return aso_max: The average maximum speed overage
    """

    overage = 0.0
    agents = list(episode_max_speeds.keys())
    num_agents = len(agents)                        # Total number of agents in system
    num_steps = len(episode_max_speeds[agents[0]])  # Total number of steps in episode

    for agent in agents:
        for step in range(num_steps):
            # max speed observed by this agent at this step
            agent_max_speed = episode_max_speeds[agent][step]

            if agent_max_speed > speed_limit:
                # Only consider speeds that are OVER the speed limit, 
                # if they are then add it to the accumulated overage
                
                overage += agent_max_speed
    
    aso_max = overage/(num_agents*num_steps)

    return aso_max


if __name__ == "__main__":

    # Get config parameters                        
    parser = MARLConfigParser()
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
                                add_system_info=True,
                                add_per_agent_info=True,
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
                                add_system_info=True,
                                add_per_agent_info=True,
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
        csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_max_speed', 'system_episode_min_max_speed', 'system_aso_max', 'system_total_stopped', 'nn_step'])    
        csv_writer.writeheader()

    # Loop over all the nn files in the nn directory
    # Every nn_save_freq steps, a new nn.pt file was saved during training
    for saved_step in range(0, args.total_timesteps, args.nn_save_freq):
        print(" > Loading network at learning step: {}".format(saved_step))

        onehot_keys = {agent: i for i, agent in enumerate(agents)}
        episode_rewards = {agent: 0 for agent in agents}            # Dictionary that maps the each agent to its cumulative reward each episode
        episode_max_speeds = {agent: [] for agent in agents}        # Dictionary that maps each agent to the maximum speed observed at each step of the agent's episode
        # episode_pressures = {agent: [] for agent in agents}         # Dictionary that maps each agent to the pressure at each step of the agent's episode (pressure = #veh leaving - #veh approaching of the intersection)

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
            next_obses, rewards, dones, truncated, info = env.step(actions)
            # info = env.unwrapped.env._compute_info()    # The wrapper class needs to be unwrapped for some reason in order to properly access info
            # print(" >>>>> INFO: {}".format(info))
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
                    # episode_pressures[agent].append(next_obses[agent][-2])  # pressure is the second to last element of the custom observation array
                
                except:
                    continue
                
                # print(" >>> episode_rewards: {}".format(episode_rewards))
                # print(" >>> rewards[{}]: {}".format(agent, rewards[agent]))
            # print(" >>>>> episode_pressures: {}".format(episode_pressures))
            obses = next_obses

            # If the simulation is done, print the episode reward and close the env
            if np.prod(list(dones.values())):
                print(" > Episode complete - logging data")

                system_episode_reward = sum(list(episode_rewards.values())) # Accumulated reward of all agents
                
                # Calculate the maximum of all max speeds observed from each agent during the episode
                agent_max_speeds = {agent:0 for agent in agents}    # max speed observed by the agent over the entire episode
                final_max_speeds = {agent:0 for agent in agents}    # last max speed observed by the agent during the episode
                # final_pressures = {agent:0 for agent in agents}     # last pressure observed by each agent in the episode
                for agent in agents:
                    agent_max_speeds[agent] = max(episode_max_speeds[agent])
                    final_max_speeds[agent] = episode_max_speeds[agent][-1]
                    # final_pressures[agent] = episode_pressures[agent][-2]

                system_episode_max_speed = max(list(agent_max_speeds.values()))
                system_episode_min_max_speed = min(list(agent_max_speeds.values()))

                # Calculate ASO max at the last step of the episode
                SPEED_LIMIT = 13.89 # TODO: config?
                # aso_max = CalculateASOMax(final_max_speeds.values(), SPEED_LIMIT, num_agents)
                aso_max = CalculateASOMax(episode_max_speeds, SPEED_LIMIT)
                print(" >> EPISODE ASO MAX: {}\n".format(aso_max))

                # Get the total number of cars stopped in the system at the end of the episode
                info = env.unwrapped.env._compute_info()    # The wrapper class needs to be unwrapped for some reason in order to properly access info
                system_total_stopped = info['agents_total_stopped']
                print( ">> TOTAL NUMBER OF STOPPED CARS IN SYSTEM: {}".format(system_total_stopped))

                # Log the episode reward to CSV
                with open(f"{csv_dir}/learning_curve.csv", "a", newline="") as csvfile:
                    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_reward', 'nn_step'])
                    csv_writer.writerow({**episode_rewards, **{'system_episode_reward': system_episode_reward, 'nn_step': saved_step}})
                
                # Log the max speeds
                with open(f"{csv_dir}/episode_max_speeds.csv", "a", newline="") as csvfile:
                    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_max_speed', 'system_episode_min_max_speed', 'system_aso_max', 'system_total_stopped', 'nn_step'])
                    csv_writer.writerow({**agent_max_speeds, **{'system_episode_max_speed': system_episode_max_speed,
                                                            'system_episode_min_max_speed': system_episode_min_max_speed,
                                                            'system_aso_max': aso_max,
                                                            'system_total_stopped': system_total_stopped,
                                                            'nn_step': saved_step}})
                
                print(" >> TOTAL EPISODE REWARD: {}\n".format(system_episode_reward))
                print(" >>> NOTE: be sure to verify the reward function being used to evaluate this model matches the reward function used to train the model")

                break
    
    env.close()

"""
sumo_analysis_batch_offline_RL.py

Description:
    Script for analyzing the policies produced by the the batch offline RL routine
    NOTE: This file currently assumes that the 4x4 sumo configuration (i.e. 16 agents) is being loaded

Usage:
    python sumo_analysis_batch_offline_RL.py -c experiments/sumo-3x3.config

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import csv
from datetime import datetime

import numpy as np

import os

# SUMO dependencies
import sumo_rl
import sys

from sumo_custom_observation import CustomObservationFunction
from sumo_custom_reward import MaxSpeedRewardFunction
from actor_critic import Actor
from offline_batch_RL_policy_learning import CalculateMaxSpeedOverage

# Config Parser
from MARLConfigParser import MARLConfigParser

# Make sure SUMO env variable is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


# Define the speed overage threshold used to evaluate the g1 constraint 
# (note this needs to match what is used in GenerateDataset)
SPEED_OVERAGE_THRESHOLD = 13.89

# Hard coded to map agents trained on the 2x2.net.xml and 2x2.rou.xml SUMO configuration
# to agents in the 4x4.net.xml and 4x4c1c2c1c2.rou.xml SUMO configuration
# Keys are the agents for the 4x4 env and values are the "trained" agents from the 2x2 env
# AGENT_ALIAS_MAP = { '0':'1',
#                     '1':'2',
#                     '2':'1',
#                     '3':'2',
#                     '4':'5',
#                     '5':'6',
#                     '6':'5',
#                     '7':'6',
#                     '8':'1',
#                     '9':'2',
#                     '10':'1',
#                     '11':'2',
#                     '12':'5',
#                     '13':'6',
#                     '14':'5',
#                     '15':'6'}

# Hard coded to map agents trained on the 2x2.net.xml and 2x2.rou.xml SUMO configuration
# to agents in the 3x3Grid2lanes.net.xml and routes14000.rou.xml SUMO configuration
AGENT_ALIAS_MAP = { '0':'1',
                    '1':'2',
                    '2':'1',
                    '3':'5',
                    '4':'6',
                    '5':'5',
                    '6':'1',
                    '7':'2',
                    '8':'1'}

if __name__ == "__main__":
    
    # Get config parameters                        
    parser = MARLConfigParser()
    args = parser.parse_args()
    
    if not args.seed:
        args.seed = int(datetime.now()) 
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    parameter_sharing_model = args.parameter_sharing_model  # Flag indicating if we're loading a model from DQN with PS
    nn_directory = args.nn_directory
    analysis_training_round = args.analysis_training_round  # Which round of batch offline RL are the policies being loaded from 
    nn_dir = f"{nn_directory}"                              # Name of directory containing the stored nn from training
    experiment_time = str(datetime.now()).split('.')[0].replace(':','-')   
    experiment_name = "{}__N{}__exp{}__seed{}__{}".format(args.gym_id, args.N, args.exp_name, args.seed, experiment_time)
    save_dir = f"batch_offline_RL_logs/{experiment_name}"
    csv_save_dir = f"{save_dir}/analysis_csv" 
    
    print(" > Parameter Sharing Enabled: {}".format(parameter_sharing_model))

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
                                add_system_info=True,
                                add_per_agent_info=True,                                
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
    onehot_keys = {agent: i for i, agent in enumerate(agents)}
    
    # Define empty dictionary that maps agents to actions
    actions = {agent: None for agent in agents}

    # Dictionary that maps the each agent to its cumulative reward each episode
    episode_rewards = {agent: 0.0 for agent in agents}            

    # Maps each agent to its MAX SPEED OVERAGE for this step        
    episode_constraint_1 = {agent : 0.0 for agent in agents}  
    
    # Maps each agent to the accumulated NUBMER OF CARS STOPPED for episode
    episode_constraint_2 = {agent : 0.0 for agent in agents}  

    print("\n=================== Environment Information ===================")
    print(" > agents: {}".format(agents))
    print(" > num_agents: {}".format(num_agents))
    print(" > action_spaces: {}".format(action_spaces))
    print(" > observation_spaces: {}".format(observation_spaces))

    # Set up the log file
    os.makedirs(csv_save_dir)
    with open(f"{csv_save_dir}/analysis_rollout.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=['step'] + 
                                                        [agent + '_accumulated_reward' for agent in agents] + 
                                                        ['total_system_reward'] +
                                                        [agent + '_accumulated_g1_return' for agent in agents] +
                                                        ['total_g1_return'] +
                                                        [agent + '_accumulated_g2_return' for agent in agents] + 
                                                        ['total_g2_return'] )
                                                        
        csv_writer.writeheader()


    # Construct the Q-Network model 
    # Note the dimensions of the model varies depending on if the parameter sharing algorithm was used or the normal independent 
    # DQN model was used
    if parameter_sharing_model:
        sys.exit(f"ERROR: Parameter sharing not yet supported")

    # Else the agents were trained using normal independent DQN so each agent gets its own Q-network model
    else: 
        
        actor_network = {}  # Dictionary for storing q-networks (maps agent to a q-network)
        
        # Load the Q-Network NN model for each agent from the specified anaylisis checkpoint step from training
        for agent in agents: 
            observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if args.global_obs else observation_spaces[agent].shape
            actor_network[agent] = Actor(observation_space_shape, action_spaces[agent].n).to(device)

            file_suffix = AGENT_ALIAS_MAP[agent]
            nn_file = "{}/policy_{}-{}.pt".format(nn_dir, analysis_training_round, file_suffix)
            print(f" > Loading policy file: {nn_file} for agent: {agent}")
            actor_network[agent].load_state_dict(torch.load(nn_file))
    

    # Seed and initialize the the environment
    for agent in agents:
        action_spaces[agent].seed(args.seed)
        observation_spaces[agent].seed(args.seed)

    obses, _ = env.reset(seed=args.seed)

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


    # Simulate the environment using actions derived from the Q-Network
    for sumo_step in range(args.sumo_seconds):
        # Populate the action dictionary
        for agent in agents:

            action, _, _ = actor_network[agent].get_action(obses[agent])
            actions[agent] = action.detach().cpu().numpy()

        # Apply all actions to the env
        next_obses, rewards, dones, truncated, info = env.step(actions)

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
                max_speed_observed_by_agent = next_obses[agent][-1]
                episode_constraint_1[agent] += CalculateMaxSpeedOverage(max_speed_observed_by_agent, SPEED_OVERAGE_THRESHOLD)
                episode_constraint_2[agent] += rewards[agent]   # NOTE That right now, the g2 constraint is the same as the 'queue' model

            except:
                continue
                
            obses = next_obses

        # Log values to csv
        with open(f"{csv_save_dir}/analysis_rollout.csv", "a", newline="") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=['step'] + 
                                                            [agent + '_accumulated_reward' for agent in agents] + 
                                                            ['total_system_reward'] +
                                                            [agent + '_accumulated_g1_return' for agent in agents] +
                                                            ['total_g1_return'] +
                                                            [agent + '_accumulated_g2_return' for agent in agents] + 
                                                            ['total_g2_return'] )
                                                            
            new_row = {}
            new_row['step'] = sumo_step
            for agent in agents:
                new_row[agent + '_accumulated_reward'] = episode_rewards[agent]
                new_row[agent + '_accumulated_g1_return'] = episode_constraint_1[agent]
                new_row[agent + '_accumulated_g2_return'] = episode_constraint_2[agent]

            new_row['total_system_reward'] = sum(episode_rewards.values())
            new_row['total_g1_return'] = sum(episode_constraint_1.values())
            new_row['total_g2_return'] = sum(episode_constraint_2.values())

            csv_writer.writerow({**new_row})


        # If the simulation is done, print the episode reward and close the env
        if np.prod(list(dones.values())):
            system_episode_reward = sum(list(episode_rewards.values())) # Accumulated reward of all agents

            print(" >> TOTAL EPISODE REWARD: {}".format(system_episode_reward))

            break
    
    env.close()

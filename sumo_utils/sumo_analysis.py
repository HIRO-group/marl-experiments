"""
sumo_analysis.py

Description:
    Script for analyzing the progress of MARL algorithms on the SUMO environment. This script is intended to load a NN model
    from a specified training checkpoint and display its performance in the SUMO GUI.

Usage:
    python sumo_analysis.py -c experiments/sumo-4x4-dqn-independent.config

"""

import torch

import numpy as np
import datetime

import os

# SUMO dependencies
import sumo_rl
import sys

from sumo_utils.sumo_custom.sumo_custom_observation import CustomObservationFunction
from sumo_utils.sumo_custom.sumo_custom_reward import CreateSumoReward
from sumo_utils.sumo_custom.calculate_speed_control import CalculateSpeedError

# Config Parser
from marl_utils.MARLConfigParser import MARLConfigParser
from rl_core.actor_critic import Actor


# Make sure SUMO env variable is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


if __name__ == "__main__":
    
    # Get config parameters                        
    parser = MARLConfigParser()
    args = parser.parse_args()
    
    # The limit used to evaluate avg speed error (the g1 metric)
    SPEED_LIMIT = args.sumo_average_speed_limit

    if not args.seed:
        args.seed = int(datetime.now()) 
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')


    analysis_steps = args.analysis_steps                    # Defines which checkpoint will be loaded into the Q model
    parameter_sharing_model = args.parameter_sharing_model  # Flag indicating if we're loading a model from DQN with PS
    nn_directory = args.nn_directory 
    nn_dir = f"{nn_directory}"                              # Name of directory containing the stored nn from training

    print(" > Parameter Sharing Enabled: {}".format(parameter_sharing_model))
    print(" > Loading NN policy from training step {} from directory: {}".format(analysis_steps, nn_directory))

    # Create the env
    # Sumo must be created using the sumo-rl module
    # NOTE: we have to use the parallel env here to conform to this implementation of dqn
    sumo_reward_function = CreateSumoReward(args=args)

    env = sumo_rl.parallel_env(net_file=args.net, 
                            route_file=args.route,
                            use_gui=args.sumo_gui,
                            max_green=args.max_green,
                            min_green=args.min_green,
                            num_seconds=args.sumo_seconds,
                            delta_time=5,
                            add_system_info=True,       # Default is True
                            add_per_agent_info=True,    # Default is True                                       
                            reward_fn=sumo_reward_function,
                            observation_class=CustomObservationFunction,
                            sumo_warnings=False)

    agents = env.possible_agents
    num_agents = len(env.possible_agents)
    # TODO: these dictionaries are deprecated, use action_space & observation_space functions instead
    action_spaces = env.action_spaces
    observation_spaces = env.observation_spaces
    onehot_keys = {agent: i for i, agent in enumerate(agents)}
    
    episode_rewards = {agent: 0 for agent in agents}        # Dictionary that maps the each agent to its cumulative reward each episode
    episode_constraint_1 = {agent: 0 for agent in agents}   # Dictionary that maps each agent to its accumulated g1 metric
    episode_constraint_2 = {agent: 0 for agent in agents}   # Dictionary that maps each agent to its accumulated g2 metric
    
    print("\n=================== Environment Information ===================")
    print(f" > agents: {agents}")
    print(f" > num_agents: {num_agents}")
    print(f" > action_spaces: {action_spaces}")
    print(f" > observation_spaces: {observation_spaces}")

    # Construct the Q-Network model 
    # NOTE: the dimensions of the model varies depending on if the parameter sharing algorithm was used or the normal independent 
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
 
        q_network = Actor(observation_space_shape, action_spaces[eg_agent].n).to(device) # In parameter sharing, all agents utilize the same q-network
        
        # Load the Q-network file
        nn_file = "{}/{}.pt".format(nn_dir, analysis_steps)
        q_network.load_state_dict(torch.load(nn_file))

    # Else the agents were trained using normal independent DQN so each agent gets its own Q-network model
    else: 
        
        q_network = {}  # Dictionary for storing q-networks (maps agent to a q-network)
        
        # Load the Q-Network NN model for each agent from the specified anaylisis checkpoint step from training
        for agent in agents: 
            observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if args.global_obs else observation_spaces[agent].shape
            q_network[agent] = Actor(observation_space_shape, action_spaces[agent].n)

            nn_file = "{}/{}-{}.pt".format(nn_dir, analysis_steps, agent) 
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
            # TODO: uncomment once the critic/actor flag is added
            # The way actions are obtained are slightly different between the actor and critic classes
            # # # if parameter_sharing_model:
            # # #     logits = q_network.forward(obses[agent].reshape((1,)+obses[agent].shape))
            # # # else:
            # # #     logits = q_network[agent].forward(obses[agent].reshape((1,)+obses[agent].shape))
            # # # actions[agent] = torch.argmax(logits, dim=1).tolist()[0]
            # Actor choses the actions
            if args.parameter_sharing_model:
                action, _, _ = q_network.get_action(obses[agent])
            else:
                action, _, _ = q_network[agent].get_action(obses[agent])

            actions[agent] = action.detach().cpu().numpy()

        # Apply all actions to the env
        next_obses, rewards, dones, truncated, info = env.step(actions)


        # If the simulation is done, print the episode reward and close the env
        if np.prod(list(dones.values())):
            system_episode_reward = sum(list(episode_rewards.values()))                         # Accumulated reward of all agents
            
            system_accumulated_g1 = sum(list(episode_constraint_1.values()))
            system_accumulated_g2 = sum(list(episode_constraint_2.values()))

            print(f" > Rollout complete after {sumo_step} steps")
            print(f"    > TOTAL EPISODE REWARD: {system_episode_reward} using reward: {args.sumo_reward}")
            print(f"    > TOTAL EPISODE g1: {system_accumulated_g1} using speed limit: {SPEED_LIMIT}")
            print(f"    > TOTAL EPISODE g2: {system_accumulated_g2}")
            for agent in agents:
                print(f"     > Agent {agent} g1: {episode_constraint_1[agent]} g2: {episode_constraint_2[agent]}")

            break

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

        # Accumulate the total episode reward and g1/g2 metrics
        for agent in agents:
            episode_rewards[agent] += rewards[agent]
            info = env.unwrapped.env._compute_info()                    # The wrapper class needs to be unwrapped for some reason in order to properly access info                
            agent_cars_stopped = info[f'{agent}_stopped']               # Get the per-agent number of stopped cars from the info dictionary
            agent_avg_speed = next_obses[agent][-2]                     # Average (true average) speed has been added to observation as second to last element

            # g1 metric
            episode_constraint_1[agent] += CalculateSpeedError(speed=agent_avg_speed, 
                                                            speed_limit=SPEED_LIMIT,
                                                            lower_speed_limit=SPEED_LIMIT)
            
            # g2 metric
            episode_constraint_2[agent] += agent_cars_stopped

        obses = next_obses

    
    env.close()

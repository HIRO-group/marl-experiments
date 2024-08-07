"""
sumo_analysis_batch_offline_RL.py

Description:
    Script for analyzing the policies produced by the the batch offline RL routine
    NOTE: This file currently assumes that the 3x3 sumo configuration (i.e. 9 agents) is being loaded 
    If parameter sharing is being used, this file is assuming that and multiple models are being evaluated 
    (specified with nn-queue-directory, nn-speed-overage-directory, and nn-speed-overage-directory-2 parameters)
    If parameter sharing is NOT being used, this file uses an AGENT_ALIAS_MAP to map agents trained on a 2x2 env
    and applies them to the 3x3 env

    Eventually these assumptions should be abstracted away to the config file

Usage:
    python sumo_analysis_batch_offline_RL.py -c experiments/sumo-3x3.config

"""

import torch
import csv
from datetime import datetime

import numpy as np

import os

# SUMO dependencies
import sumo_rl
import sys

from sumo_utils.sumo_custom.sumo_custom_observation import CustomObservationFunction
from sumo_utils.sumo_custom.sumo_custom_reward import CreateSumoReward
from sumo_utils.sumo_custom.calculate_speed_control import CalculateSpeedError
from rl_core.actor_critic import Actor

# Config Parser
from marl_utils.MARLConfigParser import MARLConfigParser

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

    # TODO: This should be moved to config
    # This is a hardcoded mapping of agents trained on the 2x2.net.xml and 2x2.rou.xml SUMO configuration
    # to agents in the 3x3.net.xml and 3x3Grid2lanes.rou.xml SUMO configuration
    # Keys are the agents for the 4x4 env and values are the "trained" agents from the 2x2 env
    # This is used for non-parameter sharing models
    AGENT_ALIAS_MAP = { '0':'1',
                        '1':'2',
                        '2':'1',
                        '3':'5',
                        '4':'6',
                        '5':'5',
                        '6':'1',
                        '7':'2',
                        '8':'1'}

    # TODO: these need to move to config once json is implemented
    # # QUEUE BASELINE
    # agent_policy_map = {'0': args.nn_queue_directory,
    #                     '1': args.nn_queue_directory,
    #                     '2': args.nn_queue_directory,
    #                     '3': args.nn_queue_directory,
    #                     '4': args.nn_queue_directory,
    #                     '5': args.nn_queue_directory,
    #                     '6': args.nn_queue_directory,
    #                     '7': args.nn_queue_directory,
    #                     '8': args.nn_queue_directory}

    # # ASL7 Baseline
    # agent_policy_map = {'0': args.nn_speed_overage_directory,
    #                     '1': args.nn_speed_overage_directory,
    #                     '2': args.nn_speed_overage_directory,
    #                     '3': args.nn_speed_overage_directory,
    #                     '4': args.nn_speed_overage_directory,
    #                     '5': args.nn_speed_overage_directory,
    #                     '6': args.nn_speed_overage_directory,
    #                     '7': args.nn_speed_overage_directory,
    #                     '8': args.nn_speed_overage_directory}
    
    # # ASL10 Baseline
    # agent_policy_map = {'0': args.nn_speed_overage_directory_2,
    #                     '1': args.nn_speed_overage_directory_2,
    #                     '2': args.nn_speed_overage_directory_2,
    #                     '3': args.nn_speed_overage_directory_2,
    #                     '4': args.nn_speed_overage_directory_2,
    #                     '5': args.nn_speed_overage_directory_2,
    #                     '6': args.nn_speed_overage_directory_2,
    #                     '7': args.nn_speed_overage_directory_2,
    #                     '8': args.nn_speed_overage_directory_2}

    # SCENARIO 1 
    # agent_policy_map = {'0': args.nn_speed_overage_directory,
    #                     '1': args.nn_speed_overage_directory,
    #                     '2': args.nn_speed_overage_directory,
    #                     '3': args.nn_speed_overage_directory,
    #                     '4': args.nn_queue_directory,
    #                     '5': args.nn_speed_overage_directory,
    #                     '6': args.nn_speed_overage_directory,
    #                     '7': args.nn_speed_overage_directory,
    #                     '8': args.nn_speed_overage_directory}

    # # Scenario 2
    # agent_policy_map = {'0': args.nn_speed_overage_directory_2,
    #                     '1': args.nn_speed_overage_directory_2,
    #                     '2': args.nn_speed_overage_directory_2,
    #                     '3': args.nn_speed_overage_directory_2,
    #                     '4': args.nn_queue_directory,
    #                     '5': args.nn_speed_overage_directory_2,
    #                     '6': args.nn_speed_overage_directory_2,
    #                     '7': args.nn_speed_overage_directory_2,
    #                     '8': args.nn_speed_overage_directory_2}

    # # Scenario 3
    # agent_policy_map = {'0': args.nn_speed_overage_directory,
    #                     '1': args.nn_speed_overage_directory_2,
    #                     '2': args.nn_speed_overage_directory,
    #                     '3': args.nn_speed_overage_directory_2,
    #                     '4': args.nn_queue_directory,
    #                     '5': args.nn_speed_overage_directory_2,
    #                     '6': args.nn_speed_overage_directory,
    #                     '7': args.nn_speed_overage_directory_2,
    #                     '8': args.nn_speed_overage_directory}
    # # Scenario 4
    # agent_policy_map = {'0': args.nn_speed_overage_directory,
    #                     '1': args.nn_speed_overage_directory,
    #                     '2': args.nn_speed_overage_directory_2,
    #                     '3': args.nn_speed_overage_directory_2,
    #                     '4': args.nn_queue_directory,
    #                     '5': args.nn_speed_overage_directory_2,
    #                     '6': args.nn_speed_overage_directory_2,
    #                     '7': args.nn_speed_overage_directory,
    #                     '8': args.nn_speed_overage_directory}

    # # Scenario 5
    # agent_policy_map = {'0': args.nn_speed_overage_directory_2,
    #                     '1': args.nn_speed_overage_directory,
    #                     '2': args.nn_speed_overage_directory_2,
    #                     '3': args.nn_speed_overage_directory,
    #                     '4': args.nn_queue_directory,
    #                     '5': args.nn_speed_overage_directory,
    #                     '6': args.nn_speed_overage_directory_2,
    #                     '7': args.nn_speed_overage_directory,
    #                     '8': args.nn_speed_overage_directory_2}

    # # Scenario 6
    # agent_policy_map = {'0': args.nn_speed_overage_directory_2,
    #                     '1': args.nn_speed_overage_directory_2,
    #                     '2': args.nn_speed_overage_directory_2,
    #                     '3': args.nn_speed_overage_directory_2,
    #                     '4': args.nn_speed_overage_directory,
    #                     '5': args.nn_speed_overage_directory_2,
    #                     '6': args.nn_speed_overage_directory_2,
    #                     '7': args.nn_speed_overage_directory_2,
    #                     '8': args.nn_speed_overage_directory_2}

    # # Scenario 7
    # agent_policy_map = {'0': args.nn_queue_directory,
    #                     '1': args.nn_queue_directory,
    #                     '2': args.nn_queue_directory,
    #                     '3': args.nn_queue_directory,
    #                     '4': args.nn_speed_overage_directory,
    #                     '5': args.nn_queue_directory,
    #                     '6': args.nn_queue_directory,
    #                     '7': args.nn_queue_directory,
    #                     '8': args.nn_queue_directory}

    # # Scenario 8
    # agent_policy_map = {'0': args.nn_speed_overage_directory_2,
    #                     '1': args.nn_queue_directory,
    #                     '2': args.nn_speed_overage_directory_2,
    #                     '3': args.nn_queue_directory,
    #                     '4': args.nn_speed_overage_directory,
    #                     '5': args.nn_queue_directory,
    #                     '6': args.nn_speed_overage_directory_2,
    #                     '7': args.nn_queue_directory,
    #                     '8': args.nn_speed_overage_directory_2}

    # # Scenario 9
    # agent_policy_map = {'0': args.nn_queue_directory,
    #                     '1': args.nn_speed_overage_directory_2,
    #                     '2': args.nn_queue_directory,
    #                     '3': args.nn_speed_overage_directory_2,
    #                     '4': args.nn_speed_overage_directory,
    #                     '5': args.nn_speed_overage_directory_2,
    #                     '6': args.nn_queue_directory,
    #                     '7': args.nn_speed_overage_directory_2,
    #                     '8': args.nn_queue_directory}

    # # Scenario 10
    # agent_policy_map = {'0': args.nn_queue_directory,
    #                     '1': args.nn_queue_directory,
    #                     '2': args.nn_speed_overage_directory_2,
    #                     '3': args.nn_speed_overage_directory_2,
    #                     '4': args.nn_speed_overage_directory,
    #                     '5': args.nn_speed_overage_directory_2,
    #                     '6': args.nn_speed_overage_directory_2,
    #                     '7': args.nn_queue_directory,
    #                     '8': args.nn_queue_directory}

    # # Scenario 11
    # agent_policy_map = {'0': args.nn_queue_directory,
    #                     '1': args.nn_queue_directory,
    #                     '2': args.nn_queue_directory,
    #                     '3': args.nn_queue_directory,
    #                     '4': args.nn_speed_overage_directory_2,
    #                     '5': args.nn_queue_directory,
    #                     '6': args.nn_queue_directory,
    #                     '7': args.nn_queue_directory,
    #                     '8': args.nn_queue_directory}

    # # Scenario 12
    # agent_policy_map = {'0': args.nn_speed_overage_directory,
    #                     '1': args.nn_speed_overage_directory,
    #                     '2': args.nn_speed_overage_directory,
    #                     '3': args.nn_speed_overage_directory,
    #                     '4': args.nn_speed_overage_directory_2,
    #                     '5': args.nn_speed_overage_directory,
    #                     '6': args.nn_speed_overage_directory,
    #                     '7': args.nn_speed_overage_directory,
    #                     '8': args.nn_speed_overage_directory}
    
    # # Scenario 13
    # agent_policy_map = {'0': args.nn_queue_directory,
    #                     '1': args.nn_speed_overage_directory,
    #                     '2': args.nn_queue_directory,
    #                     '3': args.nn_speed_overage_directory,
    #                     '4': args.nn_speed_overage_directory_2,
    #                     '5': args.nn_speed_overage_directory,
    #                     '6': args.nn_queue_directory,
    #                     '7': args.nn_speed_overage_directory,
    #                     '8': args.nn_queue_directory}

    # # Scenario 14
    # agent_policy_map = {'0': args.nn_speed_overage_directory,
    #                     '1': args.nn_queue_directory,
    #                     '2': args.nn_speed_overage_directory,
    #                     '3': args.nn_queue_directory,
    #                     '4': args.nn_speed_overage_directory_2,
    #                     '5': args.nn_queue_directory,
    #                     '6': args.nn_speed_overage_directory,
    #                     '7': args.nn_queue_directory,
    #                     '8': args.nn_speed_overage_directory}

    # Scenario 15
    agent_policy_map = {'0': args.nn_queue_directory,
                        '1': args.nn_queue_directory,
                        '2': args.nn_speed_overage_directory,
                        '3': args.nn_speed_overage_directory,
                        '4': args.nn_speed_overage_directory_2,
                        '5': args.nn_speed_overage_directory,
                        '6': args.nn_speed_overage_directory,
                        '7': args.nn_queue_directory,
                        '8': args.nn_queue_directory}


    # The limit used to evaluate avg speed error (the g1 metric)
    SPEED_LIMIT = args.sumo_average_speed_limit

    if not args.seed:
        args.seed = int(datetime.now()) 
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # Flag indicating if we're loading a model from DQN with PS
    parameter_sharing_model = args.parameter_sharing_model  
    
    # Directory containing the nn files to load
    nn_directory = args.nn_directory

    # Which round of batch offline RL are the policies being loaded from 
    analysis_training_round = args.analysis_training_round

    # If we're using a policy generated by independent DQN
    analysis_steps = args.analysis_steps

    experiment_time = str(datetime.now()).split('.')[0].replace(':','-')   
    experiment_name = "{}__N{}__exp{}__seed{}__{}".format(args.gym_id, args.N, args.exp_name, args.seed, experiment_time)
    save_dir = f"batch_offline_RL_logs/{experiment_name}"
    csv_save_dir = f"{save_dir}/analysis_csv" 
    
    print(" > Parameter Sharing Enabled: {}".format(parameter_sharing_model))

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

    # Define empty dictionary that maps agents to actions
    actions = {agent: None for agent in agents}

    # Dictionary that maps the each agent to its cumulative reward each episode
    episode_rewards = {agent: 0.0 for agent in agents}            

    # Maps each agent to its MAX SPEED OVERAGE for this step        
    episode_constraint_1 = {agent : 0.0 for agent in agents}  
    
    # Maps each agent to the accumulated NUBMER OF CARS STOPPED for episode
    episode_constraint_2 = {agent : 0.0 for agent in agents}  

    print("\n=================== Environment Information ===================")
    print(f" > agents: {agents}")
    print(f" > num_agents: {num_agents}")
    print(f" > action_spaces: {action_spaces}")
    print(f" > observation_spaces: {observation_spaces}")

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


    # Construct the Q-Network model for each agent
    actor_network = {}          
    
    # Note the dimensions of the model varies depending on if the parameter sharing algorithm was used or the normal independent 
    # DQN model was used
    if parameter_sharing_model:
        # Define the shape of the observation space depending on if we're using a global observation or not
        # Regardless, we need to add an array of length num_agents to the observation to account for one hot encoding
        eg_agent = agents[0]

        if args.global_obs:
        
            observation_space_shape = tuple((shape+1) * (num_agents) for shape in observation_spaces[eg_agent].shape)
        
        else:

            # Convert (X,) shape from tuple to int so it can be modified
            observation_space_shape = np.array(observation_spaces[eg_agent].shape).prod() + num_agents

            # Convert int to array and then to a tuple
            observation_space_shape = tuple(np.array([observation_space_shape]))                        
 
        
        for agent in agents:
            
            actor_network[agent] = Actor(observation_space_shape, action_spaces[agent].n).to(device)

            # Load the Q-network file using the provdided agent map
            agent_specific_nn_dir = agent_policy_map[agent]

            nn_file = "{}/{}.pt".format(agent_specific_nn_dir, analysis_steps)
            
            print(f" > Loading policy file: {nn_file} for agent: {agent}")
            actor_network[agent].load_state_dict(torch.load(nn_file))

    # Else the agents were trained using normal independent DQN so each agent gets its own Q-network model
    else: 
                
        # Load the Q-Network NN model for each agent from the specified anaylisis checkpoint step from training
        for agent in agents: 
            
            # TODO: break to multiple lines
            observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if args.global_obs else observation_spaces[agent].shape
            
            actor_network[agent] = Actor(observation_space_shape, action_spaces[agent].n).to(device)

            # Apply the agent mapping
            file_suffix = AGENT_ALIAS_MAP[agent]
            
            if analysis_training_round > -1:
            
                # Load the policy produced by batch offline RL algorithm
                nn_file = "{}/policy_{}-{}.pt".format(nn_directory, analysis_training_round, file_suffix)
            
            else:
            
                # Load the policy produced by the independent DQN
                nn_file = "{}/{}-{}.pt".format(nn_directory, analysis_steps, file_suffix)
            
            
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
    # NOTE: The number of steps performed may actually be less than this argument, the sumo env config arg 
    # delta_time determines how many simulated seconds pass every time 'step' is called
    for sumo_step in range(args.sumo_seconds):
        # Populate the action dictionary
        for agent in agents:

            action, _, _ = actor_network[agent].get_action(obses[agent])
            actions[agent] = action.detach().cpu().numpy()

        # Apply all actions to the env
        next_obses, rewards, dones, truncated, info = env.step(actions)

        # If the simulation is done, print the episode reward and close the env
        if np.prod(list(dones.values())):

            # Accumulated reward of all agents
            system_episode_reward = sum(list(episode_rewards.values())) 

            # Accumulated constraint values
            system_accumulated_g1 = sum(list(episode_constraint_1.values()))
            system_accumulated_g2 = sum(list(episode_constraint_2.values()))

            print(f" > Rollout complete after {sumo_step} steps")
            print(f"    > TOTAL EPISODE REWARD: {system_episode_reward} using reward: {args.sumo_reward}")
            print(f"    > TOTAL EPISODE g1: {system_accumulated_g1} using speed limit: {SPEED_LIMIT}")
            print(f"    > TOTAL EPISODE g2: {system_accumulated_g2}")

            # TODO: add per agent logging, we need to know about the performance of the center agent in terms of it's defined reward
            for agent in agents:
                print(f"      > Agent {agent} episode reward: {episode_rewards[agent]} g1: {episode_constraint_1[agent]} g2: {episode_constraint_2[agent]}")
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
        
        # Accumulate the rewards and constraints
        for agent in agents:
            episode_rewards[agent] += rewards[agent]

            # The wrapper class needs to be unwrapped for some reason in order to properly access info
            info_unwrapped = env.unwrapped.env._compute_info()

            # Get the per-agent number of stopped cars from the info dictionary
            agent_cars_stopped = info_unwrapped[f'{agent}_stopped']

            # Compute g1 metric only if there were cars present in the intersection 
            # This conforms to the way the avg speed rewards are calculated
            avg_speed_observed_by_agent = next_obses[agent][-2]

            if ((agent_cars_stopped == 0.0) and (avg_speed_observed_by_agent == 0.0)):
                
                # No cars and no average speed means there are no cars present in the intersection
                g1_from_step = 0.0

            else:

                g1_from_step = CalculateSpeedError(speed=avg_speed_observed_by_agent, 
                                                    speed_limit=SPEED_LIMIT,
                                                    lower_speed_limit=SPEED_LIMIT)

            episode_constraint_1[agent] += g1_from_step

            episode_constraint_2[agent] += agent_cars_stopped

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



    
    env.close()

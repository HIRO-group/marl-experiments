"""
sumo_generate_learning_curve.py

Description:
    File for generating the learning curve for a multi-agent model trained on the SUMO environment.
    This script evaluates the neural networks that were periodically saved during the training process
    by executing each checkpoint on the SUMO environment. The SUMO environment is configured the same way
    here as it is in the training process but here agents are not allowed to take epsilon-random actions 
    (actions are chosen soley from the model's policy). 

    NOTE: 
    This file generates logs in .\analysis\<name of nn directory>\<experiment>
    
Usage:
    python sumo_generate_learning_curve.py -c experiments/sumo-2x2-ac-independent.config    
"""

import torch

import numpy as np
from datetime import datetime

import random
import os
import csv
import pettingzoo

# SUMO dependencies
import sumo_rl
import sys
from sumo_custom.sumo_custom_observation import CustomObservationFunction
from sumo_custom.sumo_custom_reward import CreateSumoReward

# Config Parser
from marl_utils.MARLConfigParser import MARLConfigParser
from rl_core.actor_critic import Actor
from sumo_custom.calculate_speed_control import CalculateSpeedError

# Make sure SUMO env variable is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


def CalculateASOMax(episode_max_speeds, speed_limit):
    """
    TODO: function currently not used for training but update to ensure it matches reward definition
    Function for calculating The average maximum speed overage (ASOmax) after an episode,
    ASO max is essentially the average amount that each agent execeeded a given speed limit over the course of an entire episode.
    This metric is used in part to evaulate the performance of models that were trained using the "custom speed threshold" reward defined 
    for the SUMO enviornment

    Note that the speed limit here does NOT need to match the speed threshold used to train the model
    
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
    # TODO: add config flag to indicate if the Actor model or the critic model should be evaluated 
    # (each actor-critic model includes model snapshots of both the actor and critic networks)                    
    parser = MARLConfigParser()
    args = parser.parse_args()

    # TODO: config
    SPEED_LIMIT = 7.0

    # Create CSV file to store the data
    nn_directory = args.nn_directory     # TODO: just use nn_dir instead???   
    nn_dir = f"{nn_directory}"           # Name of directory where the nn was stored during training    
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
    sumo_reward_function = CreateSumoReward(args=args)

    env = sumo_rl.parallel_env(net_file=args.net, 
                            route_file=args.route,
                            use_gui=True,
                            max_green=args.max_green,
                            min_green=args.min_green,
                            num_seconds=args.sumo_seconds,
                            reward_fn=sumo_reward_function,
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
        csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_max_speed', 
                                                                'system_episode_min_max_speed', 
                                                                'system_aso_max', 
                                                                'system_accumulated_speed_overage (g1)',
                                                                'system_accumulated_queue (g2)', 
                                                                'system_total_stopped', 
                                                                'nn_step'])    
        csv_writer.writeheader()

    # Loop over all the nn files in the nn directory
    # Starting after training has started, every nn_save_freq steps, a new nn.pt file was saved during training 
    for saved_step in range(args.learning_starts + args.nn_save_freq, args.total_timesteps, args.nn_save_freq):
        print(" > Loading network at learning step: {}".format(saved_step))

        onehot_keys = {agent: i for i, agent in enumerate(agents)}
        episode_rewards = {agent: 0 for agent in agents}            # Dictionary that maps the each agent to its cumulative reward each episode
        episode_max_speeds = {agent: [] for agent in agents}        # Dictionary that maps each agent to the maximum speed observed at each step of the agent's episode
        episode_avg_speeds = {agent: [] for agent in agents}        # Dictionary that maps each agent to the avg speed observed at each step of the agent's episode
        episode_constraint_1 = {agent: 0 for agent in agents}
        episode_constraint_2 = {agent: 0 for agent in agents}

        # Construct the Q-Network model 
        # Note the dimensions of the model varies depending on if the parameter sharing algorithm was used or the normal independent DQN model was used
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
            nn_file = "{}/{}.pt".format(nn_dir, saved_step)
            q_network.load_state_dict(torch.load(nn_file))

        # Else the agents were trained using normal independent DQN so each agent gets its own Q-network model
        else: 
            
            q_network = {}  # Dictionary for storing q-networks (maps agent to a q-network)
            
            # Load the Q-Network NN model for each agent from the specified anaylisis checkpoint step from training
            for agent in agents: 
                observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if args.global_obs else observation_spaces[agent].shape
                q_network[agent] = Actor(observation_space_shape, action_spaces[agent].n).to(device) 
                nn_file = "{}/{}-{}.pt".format(nn_dir, saved_step, agent)   
                q_network[agent].load_state_dict(torch.load(nn_file))

        # Initialize the env
        print("  > Resetting environment")
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
        print("  > Executing policy from network")
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
                if parameter_sharing_model:
                    action, _, _ = q_network.get_action(obses[agent])

                else:

                    action, _, _ = q_network[agent].get_action(obses[agent])

                actions[agent] = action.detach().cpu().numpy()

            # Apply all actions to the env
            next_obses, rewards, dones, truncated, info = env.step(actions)
    
            # If the simulation is done, print the episode reward and close the env
            if np.prod(list(dones.values())):
                print("   > Episode complete - logging data")

                system_episode_reward = sum(list(episode_rewards.values())) # Accumulated reward of all agents
                
                # Calculate the maximum of all max speeds observed from each agent during the episode
                agent_max_speeds = {agent:0 for agent in agents}    # max speed observed by the agent over the entire episode

                for agent in agents:
                    agent_max_speeds[agent] = max(episode_max_speeds[agent])

                system_episode_max_speed = max(list(agent_max_speeds.values()))
                system_episode_min_max_speed = min(list(agent_max_speeds.values()))

                # Calculate ASO max at the last step of the episode
                SPEED_OVERAGE_THRESHOLD = 13.89
                aso_max = CalculateASOMax(episode_max_speeds, SPEED_OVERAGE_THRESHOLD)
                print("    > EPISODE ASO MAX: {} using speed limit of: {}".format(aso_max, SPEED_OVERAGE_THRESHOLD))

                # Get the total number of cars stopped in the system at the end of the episode
                info = env.unwrapped.env._compute_info()    # The wrapper class needs to be unwrapped for some reason in order to properly access info
                system_total_stopped = info['agents_total_stopped']
                print( "    > TOTAL NUMBER OF STOPPED CARS IN SYSTEM AT LAST STEP: {}".format(system_total_stopped))

                system_accumulated_g1 = sum(episode_constraint_1.values())
                system_accumulated_g2 = sum(episode_constraint_2.values())

                # Log the episode reward to CSV
                with open(f"{csv_dir}/learning_curve.csv", "a", newline="") as csvfile:
                    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_reward', 'nn_step'])
                    csv_writer.writerow({**episode_rewards, **{'system_episode_reward': system_episode_reward, 'nn_step': saved_step}})
                
                # Log the max speeds
                with open(f"{csv_dir}/episode_max_speeds.csv", "a", newline="") as csvfile:
                    csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['system_episode_max_speed', 
                                                                            'system_episode_min_max_speed', 
                                                                            'system_aso_max', 
                                                                            'system_accumulated_avg_speed_error (g1)', 
                                                                            'system_accumulated_queue (g2)',
                                                                            'system_total_stopped', 
                                                                            'nn_step'])
                    csv_writer.writerow({**agent_max_speeds, **{'system_episode_max_speed': system_episode_max_speed,
                                                            'system_episode_min_max_speed': system_episode_min_max_speed,
                                                            'system_aso_max': aso_max,
                                                            'system_accumulated_avg_speed_error (g1)' : system_accumulated_g1, 
                                                            'system_accumulated_queue (g2)' : system_accumulated_g2,
                                                            'system_total_stopped': system_total_stopped,
                                                            'nn_step': saved_step}})
                
                print("    > TOTAL EPISODE REWARD: {}\n".format(system_episode_reward))
                print("    > NOTE: The reward function being used to evaluate this model may not match the reward function used to train the model")

                # Go to the next policy
                break            

            # The simulation is not complete, so update the observation for the next step
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

            for agent in agents:
                episode_rewards[agent] += rewards[agent]
                
                max_speed_observed_by_agent = next_obses[agent][-1]
                episode_max_speeds[agent].append(max_speed_observed_by_agent)

                avg_speed_observed_by_agent = next_obses[agent][-2]
                episode_avg_speeds[agent].append(avg_speed_observed_by_agent)
                episode_constraint_1[agent] += CalculateSpeedError(max_speed=avg_speed_observed_by_agent, 
                                                                        speed_limit=SPEED_LIMIT,
                                                                        lower_speed_limit=SPEED_LIMIT)
                info = env.unwrapped.env._compute_info()    # The wrapper class needs to be unwrapped for some reason in order to properly access info                
                agent_cars_stopped = info[f'{agent}_stopped']   # Get the per-agent number of stopped cars from the info dictionary
                episode_constraint_2[agent] += agent_cars_stopped

            obses = next_obses

    env.close()

"""
offline_batch_RL_policy_learning.py

Description:
    Offline batch RL for learning a policy subject to constraints. The idea here is that we can utilize experiences from various
    policies to learn a new policy that is optimal according to some objective function and also obeys some secondary constraints.
    This algorithm is essentially decentralized

    NOTE: 
    This file generates logs in .\batch_offline_RL_logs\<experiment>
    
Usage:
    python offline_batch_RL_policy_learning.py -c experiments/sumo-2x2-ac-independent.config

References:
    https://arxiv.org/pdf/1903.08738.pdf

"""

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from datetime import datetime
import random
import csv
import pickle

# SUMO dependencies
import sumo_rl
import sys
import os
from sumo_custom_observation import CustomObservationFunction
from sumo_custom_reward import MaxSpeedRewardFunction

# Config Parser
from MARLConfigParser import MARLConfigParser

# Custom modules
from actor_critic import Actor, QNetwork, one_hot_q_values
from dataset import Dataset
from calculate_speed_control import CalculateMaxSpeedOverage

# Set up the system and environement
# Make sure SUMO env variable is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


def GenerateDataset(env: sumo_rl.parallel_env,
                    list_of_policies: list,
                    exceess_speed_action_ratio:float = 0.4,
                    queue_action_ratio:float = 0.4, 
                    num_episodes:int=100, 
                    episode_steps:int=1000,
                    parameter_sharing_model:bool=False) -> dict:
    """
    :param env: The sumo environment
    :param list_of_policies: A list of trained neural networks used to generate the dataset by acting in the environment
        NOTE: This function assumes that the list of policies is ordered like [excess_speed_model, queue_model]
    :param excess_speed_action_ratio: The fraction of actions that should come from the queue policy when generating  
            the dataset, if excess_speed_action_ratio + queue_action_ratio are less than 1, the remaining fraction of actions
            will come from a completely random policy e.g. if 0.4 all actions come from "excess speed" and 0.4 come from
            "queue" then 0.2 of all actions will be random
    :param queue_action_ratio: The fraction of actions that should come from the queue policy when generating the 
            dataset, if excess_speed_action_ratio + queue_action_ratio are less than 1, the remaining fraction of actions
            will come from a completely random policy e.g. if 0.4 all actions come from "excess speed" and 0.4 come from
            "queue" then 0.2 of all actions will be random
    :param num_episodes: number of episodes to run to populate the dataset
    :param episode_steps: number of steps to take in each episode
    :param parameter_sharing_model: Flag indicating if parameter sharing was used to generate the policies that are
            being used to generate the dataset, if yes, each agent will still get its own unique dataset
    :returns a dictionary that maps each agent to 
    """
    print(f"  >> Generating dataset with {num_episodes} episodes of {episode_steps} steps")
    start_time = datetime.now()

    DATASET_SIZE = num_episodes*episode_steps
    SPEED_OVERAGE_THRESHOLD = 13.89 # TODO: config
    # SPEED_LOWER_THRESHOLD = 0.01
    # SPEED_LOWER_THRESHOLD = 5.0
    # SPEED_LOWER_THRESHOLD = 0.0
    SPEED_LOWER_THRESHOLD = 1.0
    
    agents = env.possible_agents
    num_agents = len(agents)
    optimal_action_ratio = exceess_speed_action_ratio + queue_action_ratio  # Ratio of optimal actions that should be in this dataset

    onehot_keys = {agent: i for i, agent in enumerate(agents)}

    # Initialize the dataset as a dictionary that maps agents to Dataset objects that are full of experience
    dataset = {agent : Dataset(DATASET_SIZE) for agent in agents}

    # Define empty dictionary tha maps agents to actions
    actions = {agent: None for agent in agents}
    action_spaces = env.action_spaces

    # Define dictionaries to hold the values of the constraints (g1 and g2) each step
    constraint_1 = {agent : 0 for agent in agents}  # Maps each agent to its (-1) * MAX SPEED OVERAGE for this step
    constraint_2 = {agent : 0 for agent in agents}  # Maps each agent to the (-1) * NUBMER OF CARS STOPPED for this step

    for episode in range(num_episodes):
        print(f"   >>> Generating episode: {episode}")

        # Reset the environment
        obses, _ = env.reset()

        if parameter_sharing_model:
            # If parameter sharing is being used, one-hot encoding must be applied to the observations to match the 
            # dimensions of each agent's policy network

            for agent in agents:
                onehot = np.zeros(num_agents)
                onehot[onehot_keys[agent]] = 1.0
                obses[agent] = np.hstack([onehot, obses[agent]])            

        for step in range(episode_steps):

            # Set the action for each agent
            for agent in agents:
    
                # Select which policy to use according to the provided action ratios
                q_network = np.random.choice(['random'] + list_of_policies, 1, p=[1-optimal_action_ratio, exceess_speed_action_ratio, queue_action_ratio])[0]

                if (q_network == 'random'):
                    # Use a random action
                    actions[agent] = action_spaces[agent].sample()

                else:
                    # Actor choses the actions
                    action, _, _ = q_network[agent].to(device).get_action(obses[agent])
                    actions[agent] = action.detach().cpu().numpy()

            # Apply all actions to the env
            next_obses, rewards, dones, truncated, info = env.step(actions)
            
            if np.prod(list(dones.values())):
                # Start the next episode
                print(f"   >>> Episode complete at {step} steps, going to next episode")
                break

            if parameter_sharing_model:
                for agent in agents:
                    onehot = np.zeros(num_agents)
                    onehot[onehot_keys[agent]] = 1.0
                    next_obses[agent] = np.hstack([onehot, next_obses[agent]])

            # Caclulate constraints and add the experience to the dataset
            for agent in agents:
                 max_speed_observed_by_agent = next_obses[agent][-1]
                 constraint_1[agent] = CalculateMaxSpeedOverage(max_speed=max_speed_observed_by_agent, 
                                                                speed_limit=SPEED_OVERAGE_THRESHOLD,
                                                                lower_speed_limit=SPEED_LOWER_THRESHOLD)
                 constraint_2[agent] = rewards[agent]   # NOTE: This assumes that the environment was configured with the "queue" reward
                 dataset[agent].put((obses[agent], actions[agent], next_obses[agent], constraint_1[agent], constraint_2[agent], dones[agent]))

            obses = next_obses


    stop_time = datetime.now()
    print(" >> Dataset generation complete")
    print(" >> Total execution time: {}".format(stop_time-start_time))

    env.close()

    return dataset

def CalculateAverageRewardPerStep(queue_length_env:sumo_rl.parallel_env,
                                  policies_to_use:dict,
                                  reward_to_evaluate:str,
                                  config_args) -> float:
    """
    Run a rollout episode in the SUMO environment using a provided policy, calcualte and return a metric averaged per step across
    all agents
    # TODO: If a dataset was not provided and we have to generate a new dataset, this function should be part of that step

    :param queue_length_env: The SUMO environment to execute the policy in, assumes that the reward has been set to "queue"
    :param policies_to_use: Dictionary that maps agents to "actor" models to use for the evaluation
    :param reward_to_evaluate: Either "queue" or "speed_overage", determines which metric to compute during the episode
    :param config_args: Configuration arguments used to set up the experiment
    :returns: Either average "queue" reward per step (averaged for all agents) or average "speed overage" reward per step 
            (averaged for all agents)
    """
    # TODO: update function to support global observations
    
    # Determine if a proper reward evaluation was requested
    if (reward_to_evaluate != 'speed_overage') and (reward_to_evaluate != 'queue'):
        print(f"  > ERROR: Unrecognized reward evaluation '{reward_to_evaluate}' requested")
        print(f"  > Function only supports 'speed_overage' or 'queue'")
        return 0.0
    

    # Define the speed overage threshold used to evaluate the g1 constraint 
    # (note this needs to match what is used in GenerateDataset)
    SPEED_OVERAGE_THRESHOLD = 13.89
    # SPEED_LOWER_THRESHOLD = 0.01
    # SPEED_LOWER_THRESHOLD = 5.0
    # SPEED_LOWER_THRESHOLD = 0.0
    SPEED_LOWER_THRESHOLD = 1.0 

    agents = queue_length_env.possible_agents

    # Define empty dictionary that maps agents to actions
    actions = {agent: None for agent in agents}

    # Dictionary that maps the each agent to its cumulative reward each episode
    episode_rewards = {agent: 0.0 for agent in agents}            

    # Maps each agent to its MAX SPEED OVERAGE for this step        
    episode_constraint_1 = {agent : 0.0 for agent in agents}  
    
    # Maps each agent to the accumulated NUBMER OF CARS STOPPED for episode
    episode_constraint_2 = {agent : 0.0 for agent in agents}  

    # Initialize the env
    obses, _ = queue_length_env.reset()

    if config_args.parameter_sharing_model:
        # Apply one-hot encoding to the initial observations
        onehot_keys = {agent: i for i, agent in enumerate(agents)}

        for agent in agents:
            onehot = np.zeros(num_agents)
            onehot[onehot_keys[agent]] = 1.0
            obses[agent] = np.hstack([onehot, obses[agent]])

    # Perform the rollout
    for sumo_step in range(config_args.sumo_seconds):
        # Populate the action dictionary
        for agent in agents:
            # Only use optimal actions according to the policy
            action, _, _ = policies_to_use[agent].to(device).get_action(obses[agent])
            actions[agent] = action.detach().cpu().numpy()

        # Apply all actions to the env
        next_obses, rewards, dones, truncated, info = queue_length_env.step(actions)

        if np.prod(list(dones.values())):
            break

        if config_args.parameter_sharing_model:
            # Apply one-hot encoding to the observations
            onehot_keys = {agent: i for i, agent in enumerate(agents)}

            for agent in agents:
                onehot = np.zeros(num_agents)
                onehot[onehot_keys[agent]] = 1.0
                next_obses[agent] = np.hstack([onehot, next_obses[agent]])

        # Accumulate the total episode reward and max speeds
        for agent in agents:
            # At the end of a simulation, next_obses is an empty dictionary so don't log it 
            episode_rewards[agent] += rewards[agent]        # NOTE This should be the same as episode_constraint_2 if the env was configured correctly
            max_speed_observed_by_agent = next_obses[agent][-1]
            episode_constraint_1[agent] += CalculateMaxSpeedOverage(max_speed=max_speed_observed_by_agent, 
                                                                    speed_limit=SPEED_OVERAGE_THRESHOLD,
                                                                    lower_speed_limit=SPEED_LOWER_THRESHOLD)
            episode_constraint_2[agent] += rewards[agent]   # NOTE That right now, the g2 constraint is the same as the 'queue' model
                
        obses = next_obses
    print(f"   > Rollout complete after {sumo_step} steps")
    print(f"    > Constraint 1 total return: {sum(episode_constraint_1.values())}")
    print(f"    > Constraint 2 total return: {sum(episode_constraint_2.values())}")
    for agent in agents: 
        print(f"      > Agent '{agent}' constraint 1 return: {episode_constraint_1[agent]}")
        print(f"      > Agent '{agent}' constraint 2 return: {episode_constraint_2[agent]}")

    if (reward_to_evaluate == 'speed_overage'):
        # Average value per step for each agent
        avg_g1s_per_step = [agent_g1_total_returns/sumo_step for agent_g1_total_returns in episode_constraint_1.values()]
        print(f"      > Average g1s per step for each agent: \n{avg_g1s_per_step}")
        # Average value per step averaged across all agents
        avg_g1_per_step_per_agent = np.mean(avg_g1s_per_step)

        return avg_g1_per_step_per_agent

    elif (reward_to_evaluate == 'queue'):
        # Average value per step for each agent
        avg_g2s_per_step = [agent_g2_total_returns/sumo_step for agent_g2_total_returns in episode_constraint_2.values()]
        print(f"      > Average g2s per step for each agent: \n{avg_g2s_per_step}")
        # Average value per step averaged across all agents
        avg_g2_per_step_per_agent = np.mean(avg_g2s_per_step)

        return avg_g2_per_step_per_agent


def NormalizeDataset(dataset:dict,
                     constraint_ratio:float,
                     g1_upper_bound:float,
                     g1_lower_bound:float) -> dict:
    """
    Function for normalizing the dataset so that the values of one constraint do not normalize the other
    :param dataset: Dictionary that maps agents to a dataset of experience tuples
    :param constraint_ratio: The ratio to use in the weight adjustment, used to determine where the g1 constraint 
            should be applied between the upper and lower bounds
    :param g1_upper_bound: Upper bound to apply to g1 values in the dataset, should be the average reward per step 
            (avg of all agents) when evaluating the excess speed policy according to the excess speed reward
    :param g2_lower_bound: Lower bound to apply to g1 values in the dataset, should be the average reward per step 
            (avg of all agents) when evaluating the excess speed policy according to the excess speed reward

    :returns Dictionary that maps agents to normalized datasets
    """
    # # # print(f" > Normalizing dataset constraint values")
    # # # # Intialize dataset
    # # # normalized_dataset = {}

    # # # total_c_g1 = 0.0
    # # # total_c_g2 = 0.0

    # # # for agent in dataset.keys():
        
    # # #     # Initialize constraint bounds
    # # #     C_g1 = 0.0
    # # #     C_g2 = 0.0
        
    # # #     G_1 = 0.0
    # # #     G_2 = 0.0

    # # #     normalized_g1s = []
    # # #     normalized_g2s = []

    # # #     # Number of experience tuples for this agent
    # # #     n = len(dataset[agent].buffer)
    # # #     print(f"  >> Agent '{agent}' buffer size: {n} ")

    # # #     normalized_dataset[agent] = Dataset(n)
        
    # # #     # Calculate the bounds for the g1 and g2 constraints
    # # #     for i in range(n):
    # # #         s_obses, s_actions, s_next_obses, s_g1s, s_g2s, s_dones = dataset[agent].buffer[i]
    # # #         C_g1 += s_g1s
    # # #         C_g2 += s_g2s

    # # #     C_g1 = C_g1 / n * (1.0 + constraint_ratio)
    # # #     # C_g2 = C_g2 / n * (1.0 - constraint_ratio)    

    # # #     # Now apply the bounds to the g1 and g2 observations
    # # #     for i in range(n):
    # # #         s_obses, s_actions, s_next_obses, s_g1s, s_g2s, s_dones = dataset[agent].buffer[i]
            
    # # #         g1_normalized = min(C_g1, s_g1s) - C_g1
    # # #         normalized_g1s.append(g1_normalized)
    # # #         G_1 += g1_normalized

    # # #         # g2_normalized = min(C_g2, s_g2s) - C_g2   
    # # #         g2_normalized = s_g2s # No constraint for g2
    # # #         normalized_g2s.append(g2_normalized)
    # # #         G_2 += g2_normalized
            
    # # #     for i in range(n):
    # # #         s_obses, s_actions, s_next_obses, _, _, s_dones = dataset[agent].buffer[i]
            
    # # #         # Calculate normalized g1 and g2
    # # #         # NOTE: weirdly, the best response we have seen (both g1 and g2 reaching maximal returns) occured 
    # # #         # without this step... 
    # # #         # Not sure why
    # # #         g1_n = normalized_g1s[i]/abs(G_1)
    # # #         if g1_n > 0.0:
    # # #             print(f" > ERROR: normalized g1 = {g1_n}, algorithm assumes g1 < 0")
    # # #             sys.exit()

    # # #         g2_n = normalized_g2s[i]/abs(G_2)
    # # #         if g2_n > 0.0:
    # # #             print(f" > ERROR: normalized g2 = {g2_n}, algorithm assumes g2 < 0")
    # # #             sys.exit()
            

    # # #         # Add it to the new dataset
    # # #         normalized_dataset[agent].put((s_obses, s_actions, s_next_obses, g1_n, g2_n, s_dones))

    # # #     total_c_g1 += C_g1
    # # #     total_c_g2 += C_g2
    # # #     print(f"  >> Agent: {agent}")
    # # #     print(f"   >>> C_g1 = {C_g1}")
    # # #     print(f"   >>> C_g2 = {C_g2}")

    # # #     print(f"   >>> G_1 = {G_1}")
    # # #     print(f"   >>> G_2 = {G_2}")

    # # # # These can be thought of as the constraints to show on the plots from online rollouts
    # # # print(f" > total_c_g1 = {total_c_g1}")  
    # # # print(f" > total_c_g2 = {total_c_g2}")


    print(f" > Normalizing dataset constraint values")
    normalized_dataset = {}

    total_g1 = 0.0
    total_g2 = 0.0

    # Calculate constraint to be applied to g1 returns
    c1 = ((g1_upper_bound - g1_lower_bound) * constraint_ratio) + g1_lower_bound
    print(f" > Constraint for g1 (c1) = {c1}")  

    for agent in dataset.keys():

        G_1 = 0.0
        G_2 = 0.0

        adjusted_g1s = []
        adjusted_g2s = []

        # Number of experience tuples for this agent
        n = len(dataset[agent].buffer)
        print(f"   > Agent '{agent}' buffer size: {n} ")

        normalized_dataset[agent] = Dataset(n)

        for i in range(n):
            s_obses, s_actions, s_next_obses, s_g1s, s_g2s, s_dones = dataset[agent].buffer[i]
            
            # Apply constraint to g1
            g1 = min(c1, s_g1s) - c1

            adjusted_g1s.append(g1)
            G_1 += g1

            adjusted_g2s.append(s_g2s)
            G_2 += s_g2s


        for i in range(n):
            s_obses, s_actions, s_next_obses, _, _, s_dones = dataset[agent].buffer[i]

            # Calculate normalized g1 and g2
            # NOTE: weirdly, the best response we have seen (both g1 and g2 reaching maximal returns) occured
            # without this step...
            # Not sure why
            g1_n = adjusted_g1s[i]/abs(G_1)
            if (g1_n > 0.0):
                print(f" > ERROR: normalized g1 = {g1_n}, algorithm assumes g1 < 0")
                sys.exit()

            g2_n = adjusted_g2s[i]/abs(G_2)
            if (g2_n > 0.0):
                print(f" > ERROR: normalized g2 = {g2_n}, algorithm assumes g2 < 0")
                sys.exit()
            

            # Add it to the new dataset
            normalized_dataset[agent].put((s_obses, s_actions, s_next_obses, g1_n, g2_n, s_dones))

        total_g1 += G_1
        total_g2 += G_2
        print(f"  > Agent: {agent}")

        print(f"    > G_1 = {G_1}")
        print(f"    > G_2 = {G_2}")

    print(f" > total_g1 = {total_g1}")  
    print(f" > total_g2 = {total_g2}")            

    return normalized_dataset


def OfflineBatchRL(env:sumo_rl.parallel_env,
                    dataset: dict,
                    dataset_policies:list,
                    config_args,
                    nn_save_dir:str,
                    csv_save_dir:str,
                    max_num_rounds:int=10) -> tuple[dict, dict, list, list]:
    """
    Perform offline batch reinforcement learning
    Here we use a provided dataset to learn and evaluate a policy for a given number of "rounds"
    Each round, a policy is learned and then evaluated (each of which involves solving an RL problem).
    The provided constraint function defines how the "target" is calculated for each RL problem. At the end of the
    round, the expected value of the polciy and the value function is calculated.

    :param env: The SUMO environment that was used to generate the dataset
    :param dataset: Dictionary that maps each agent to its experience tuple
    :param dataset_policies: List of policies that were used to generate the dataset for this experiment (used for online evaluation)
    :param config_args: Configuration arguments used to set up the experiment
    :param nn_save_dir: Directory in which to save the models each round
    :pram csv_save_dir: Directory in which to save the csv file
    :param max_num_rounds: The number of rounds to perform (T)
    :returns A dictionary that maps each agent to its mean learned policy,
            a dictionary that maps each agent to the last learned policy,
            a list of the final values for mean lambda 1 and mean lambda 2,
            a list of the final values of lambda 1 and lambda 2
    """

    print(f" > Performing batch offline reinforcement learning")
    function_start_time = datetime.now()

    # NOTE: These are the raw objects from the environment, they should not be modified for parameter sharing
    # Dimensions of the observation space can be modified within sub-functions where necessary
    agents = env.possible_agents
    action_spaces = env.action_spaces
    observation_spaces = env.observation_spaces

    # There are two mean constraint value functions to track so create some strings to append to the corresponding file names for each constraint
    mean_constraint_suffixes = ['g1', 'g2']


    # Initialize csv files
    with open(f"{csv_save_dir}/FQE_loss_{mean_constraint_suffixes[0]}.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['Q(s,a) Sample','global_step'])
        csv_writer.writeheader()

    with open(f"{csv_save_dir}/FQE_loss_{mean_constraint_suffixes[1]}.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['Q(s,a) Sample','global_step'])
        csv_writer.writeheader()

    with open(f"{csv_save_dir}/FQI_actor_loss.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['Pi(a|s) Sample', 'global_step'])
        csv_writer.writeheader()

    with open(f"{csv_save_dir}/mean_policy_loss.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['global_step', 'round'])
        csv_writer.writeheader()

    with open(f"{csv_save_dir}/mean_constraint_loss_{mean_constraint_suffixes[0]}.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['global_step', 'round'])
        csv_writer.writeheader()

    with open(f"{csv_save_dir}/mean_constraint_loss_{mean_constraint_suffixes[1]}.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['global_step', 'round'])
        csv_writer.writeheader()

    with open(f"{csv_save_dir}/rollouts.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=['round'] + [agent + '_return_g1' for agent in agents] +
                                                        ['system_return_g1'] +
                                                        [agent + '_return_g2' for agent in agents] +
                                                        ['system_return_g2'] +
                                                        ['lambda_1', 'lambda_2', 'mean_lambda_1', 'mean_lambda_2'])
        csv_writer.writeheader()


    with open(f"{csv_save_dir}/online_rollouts.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=['round'] +
                                                        [agent + '_mean_policy_g1_return' for agent in agents] +
                                                        ['mean_policy_system_return_g1'] +
                                                        [agent + '_mean_policy_g2_return' for agent in agents] +
                                                        ['mean_policy_system_return_g2'] +

                                                        [agent + '_current_policy_g1_return' for agent in agents] +
                                                        ['current_policy_system_return_g1'] +
                                                        [agent + '_current_policy_g2_return' for agent in agents] +
                                                        ['current_policy_system_return_g2'] +

                                                        [agent + '_threshold_policy_g1_return' for agent in agents] +
                                                        ['threshold_policy_system_return_g1'] +
                                                        [agent + '_threshold_policy_g2_return' for agent in agents] +
                                                        ['threshold_policy_system_return_g2'] +

                                                        [agent + '_queue_policy_g1_return' for agent in agents] +
                                                        ['queue_policy_system_return_g1'] +
                                                        [agent + '_queue_policy_g2_return' for agent in agents] +
                                                        ['queue_policy_system_return_g2'])
        csv_writer.writeheader()

    # Define a "mini" dataset to be used for offline rollouts (basically this will get passed through networks to evaluate them)
    rollout_mini_dataset = {agent: dataset[agent].sample(50000) for agent in agents}    # TODO: config, currently set to the same size as the dataset itself

    # Define an "example" agent that can be used as a dictionary key when the specific agent doesn't matter
    eg_agent = agents[0]    
    prev_mean_policies = {}
    prev_g1_constraints = {}
    prev_g2_constraints = {}

    print(f"  > Initializing neural networks")
    num_agents = len(agents) 

    if config_args.parameter_sharing_model:
        # Modify the observation space shape to include one-hot-encoding used for parameter sharing
        print(f"   > Parameter sharing enabled")

        # onehot_keys = {agent: i for i, agent in enumerate(agents)}
        if config_args.global_obs:
            print(f"    > Global observations enabled")
            observation_space_shape = tuple((shape+1) * (num_agents) for shape in observation_spaces[eg_agent].shape)

        else:
            print(f"    > Global observations NOT enabled")
            observation_space_shape = np.array(observation_spaces[eg_agent].shape).prod() + num_agents


    else:
        # Only need to modify the observation space shape if global observations are being used
        print(f"   > Parameter sharing NOT enabled")

        if config_args.global_obs:
            print(f"    > Global observations enabled")
            observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape)
    
        else:
            print(f"    > Global observations NOT enabled")
            observation_space_shape = observation_spaces[eg_agent].shape
    
    # NOTE: When using parameter sharing, one network is needed for each object (policy and constraints). In this case, one network (per object) will be 
    # created and trained but to conform to the other functions, each agent will get it's own copy of that network (via dictionary)
    #            
    # When not using parameter sharing, each agent gets its own unique policy and constraint networks and these will be trained independently
    for agent in agents:
        prev_mean_policies[agent] = Actor(observation_space_shape, action_spaces[agent].n).to(device)
        prev_g1_constraints[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
        prev_g2_constraints[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)

    # Initialize both lambdas equally
    lambda_1 = 1.0/2.0
    lambda_2 = 1.0/2.0
    mean_lambda_1 = lambda_1
    mean_lambda_2 = lambda_2

    # for t=1:T
    for t in range(1,max_num_rounds+1):
        print(f" >> BEGINNING ROUND: {t} OF {max_num_rounds}")
        round_start_time = datetime.now()
        # Learn a policy that optimizes actions for the weighted sum of the g1 and g2 constraints
        # This is essentially the "actor", policies here are represented as probability density functions of taking an action given a state
        # NOTE: Regardless if parameter sharing is being used or not, the output here is a dictionary that maps each agent to its policy network
        # when parameter sharing is being used, each network is identical (that does not mean each agent has identical policies)
        policies = FittedQIteration(observation_spaces, 
                                    action_spaces, 
                                    agents,
                                    dataset,
                                    csv_save_dir,
                                    config_args,
                                    constraint="weighted_sum",
                                    lambda_1=lambda_1,
                                    lambda_2=lambda_2) 
        
        # Save the policy every round
        # Format is policy_<round>-<agent>.pt
        for a in agents:
            torch.save(policies[a].state_dict(), f"{nn_save_dir}/policies/policy_{t}-{a}.pt")

        # Evaluate the constraint value functions (these are essentially the "critics")
        # Evaluate G_1^pi (the speed overage constraint)
        # NOTE: Regardless if parameter sharing is being used or not, the output here is a dictionary that maps each agent to its constraint network
        # when parameter sharing is being used, each network is identical (that does not mean each agent has identical constraint estimates)            
        G1_pi = FittedQEvaluation(observation_spaces, 
                                    action_spaces, 
                                    agents,
                                    policies,
                                    dataset,
                                    csv_save_dir,
                                    mean_constraint_suffixes[0],    # Assumes order was [g1, g2]
                                    config_args, 
                                    constraint="speed_overage")

        # Save the value function every round
        # Format is constraint_<round>-<agent>.pt
        for a in agents:
            torch.save(G1_pi[a].state_dict(), f"{nn_save_dir}/constraints/speed_overage/constraint_{t}-{a}.pt")


        # Evaluate G_2^pi 
        # NOTE: Regardless if parameter sharing is being used or not, the output here is a dictionary that maps each agent to its constraint network
        # when parameter sharing is being used, each network is identical (that does not mean each agent has identical constraint estimates) 
        G2_pi = FittedQEvaluation(observation_spaces, 
                                    action_spaces, 
                                    agents,
                                    policies,
                                    dataset,
                                    csv_save_dir,
                                    mean_constraint_suffixes[1],    # Assumes order was [g1, g2]
                                    config_args, 
                                    constraint="queue")

        # Save the value function every round
        # Format is constraint_<round>-<agent>.pt
        for a in agents:
            torch.save(G2_pi[a].state_dict(), f"{nn_save_dir}/constraints/queue/constraint_{t}-{a}.pt")


        # Calculate 1/t*(pi + t-1(E[pi])) for each agent
        mean_policies = CalculateMeanPolicy(policies,
                                            prev_mean_policies,
                                            observation_spaces,
                                            action_spaces,
                                            agents,
                                            t,
                                            dataset,
                                            csv_save_dir,
                                            config_args)

        # Save the mean policy each round
        # Format is policy_<round>-<agent>.pt
        for a in agents:
            torch.save(mean_policies[a].state_dict(), f"{nn_save_dir}/policies/mean/policy_{t}-{a}.pt")

        # Calculate 1/t*(g1 + t-1(E[g1])) for each agent
        mean_g1_constraints = CalculateMeanConstraint(G1_pi,
                                                      prev_g1_constraints,
                                                      observation_spaces,
                                                      action_spaces,
                                                      agents,
                                                      t,
                                                      dataset,
                                                      csv_save_dir,
                                                      mean_constraint_suffixes[0],  # Assumes order was [g1, g2]
                                                      config_args)

        # Save the mean value function each round
        # Format is constraint_<round>-<agent>.pt
        for a in agents:
            torch.save(mean_g1_constraints[a].state_dict(), f"{nn_save_dir}/constraints/speed_overage/mean/constraint_{t}-{a}.pt")


        # Calculate 1/t*(g2 + t-1(E[g2])) for each agent
        mean_g2_constraints = CalculateMeanConstraint(G2_pi,
                                                      prev_g2_constraints,
                                                      observation_spaces,
                                                      action_spaces,
                                                      agents,
                                                      t,
                                                      dataset,
                                                      csv_save_dir,
                                                      mean_constraint_suffixes[1],  # Assumes order was [g1, g2]
                                                      config_args)

        # Save the mean value function each round
        # Format is constraint_<round>-<agent>.pt
        for a in agents:
            torch.save(mean_g2_constraints[a].state_dict(), f"{nn_save_dir}/constraints/queue/mean/constraint_{t}-{a}.pt")

        # Update mean networks for the next round
        prev_mean_policies = mean_policies
        prev_g2_constraints = mean_g2_constraints
        prev_g1_constraints = mean_g1_constraints


        # Perform offline rollouts using each value function and a small portion of the provided dataset
        # NOTE: The returns here are dictionaries that map agents to their return
        print(f" > EVALUATING G1_pi IN OFFLINE ROLLOUT")
        g1_returns = PerformOfflineRollout(G1_pi, policies, rollout_mini_dataset)

        print(f" > EVALUATING G2_pi IN OFFLINE ROLLOUT")
        g2_returns = PerformOfflineRollout(G2_pi, policies, rollout_mini_dataset)
        # TODO: perform some kind of similar rollout for the dataset policies to compare?


        # DEBUGGING:
        offline_g1_returns_threshold_policy = PerformOfflineRollout(G1_pi, dataset_policies[0], rollout_mini_dataset)
        offline_g2_returns_threshold_policy = PerformOfflineRollout(G2_pi, dataset_policies[0], rollout_mini_dataset)
        offline_g1_returns_queue_policy = PerformOfflineRollout(G1_pi, dataset_policies[1], rollout_mini_dataset)
        offline_g2_returns_queue_policy = PerformOfflineRollout(G2_pi, dataset_policies[1], rollout_mini_dataset)
        print(f" > OFFLINE ROLLOUT RESULTS:")
        print(f"   > CURRENT POLICY G1_pi: {torch.sum(torch.tensor(list(g1_returns.values()))).detach().numpy()} ")
        print(f"   > CURRENT POLICY G2_pi: {torch.sum(torch.tensor(list(g2_returns.values()))).detach().numpy()} ")
        print(f"   > THRESHOLD POLICY G1_pi: {torch.sum(torch.tensor(list(offline_g1_returns_threshold_policy.values()))).detach().numpy()} ")
        print(f"   > THRESHOLD POLICY G2_pi: {torch.sum(torch.tensor(list(offline_g2_returns_threshold_policy.values()))).detach().numpy()} ")
        print(f"   > QUEUE POLICY G1_pi: {torch.sum(torch.tensor(list(offline_g1_returns_queue_policy.values()))).detach().numpy()} ")
        print(f"   > QUEUE POLICY G2_pi: {torch.sum(torch.tensor(list(offline_g2_returns_queue_policy.values()))).detach().numpy()} ")

        # NOTE: We are logging the lambda values produced by round X rather than the values used in
        # round X
        # Adjust lambda
        if (t == 1):
            lambda_1, lambda_2, R_1, R_2 = OnlineLambdaLearning(lambda_1, lambda_2, g1_returns, g2_returns)
        else:
            lambda_1, lambda_2, R_1, R_2 = OnlineLambdaLearningByImprovementRate(lambda_1, lambda_2, g1_returns, g2_returns, R_1, R_2)


        if (t > 1):
            # On the first round, the expected value of lambda was set as the initial value of lambda
            mean_lambda_1 = 1/t*(lambda_1 + ((t-1) * mean_lambda_1))
            mean_lambda_2 = 1/t*(lambda_2 + ((t-1) * mean_lambda_2))

        # Save the rollout returns and the updated lambdas
        with open(f"{csv_save_dir}/rollouts.csv", "a", newline="") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=['round'] + 
                                                            [agent + '_return_g1' for agent in agents] + 
                                                            ['system_return_g1'] + 
                                                            [agent + '_return_g2' for agent in agents] + 
                                                            ['system_return_g2'] +
                                                            ['lambda_1', 'lambda_2', 'mean_lambda_1', 'mean_lambda_2'])
            new_row = {}
            new_row['round'] = t
            for agent in agents:
                new_row[agent + '_return_g1'] = g1_returns[agent].item()
            new_row['system_return_g1'] = torch.sum(torch.tensor(list(g1_returns.values()))).detach().numpy()
            for agent in agents:
                new_row[agent + '_return_g2'] = g2_returns[agent].item()
            new_row['system_return_g2'] = torch.sum(torch.tensor(list(g2_returns.values()))).detach().numpy()
            new_row['lambda_1'] = lambda_1
            new_row['lambda_2'] = lambda_2
            new_row['mean_lambda_1'] = mean_lambda_1
            new_row['mean_lambda_2'] = mean_lambda_2

            csv_writer.writerow({**new_row})

        # Now run some online rollouts to compare performance between the learned policy and the dataset policies
        print(f"  > Performing online rollout for mean policy")
        _, mean_policy_g1_return, mean_policy_g2_return = PerformRollout(env, prev_mean_policies, config_args)
        
        print(f"  > Performing online rollout for current learned policy")
        _, current_policy_g1_return, current_policy_g2_return = PerformRollout(env, policies, config_args)

        # TODO: make this more generic
        threshold_policy = dataset_policies[0]
        print(f"  > Performing online rollout for speed overage policy")
        _, threshold_policy_g1_return, threshold_policy_g2_return = PerformRollout(env, threshold_policy, config_args)

        queue_policy = dataset_policies[1]
        print(f"  > Performing online rollout for queue policy")
        _, queue_policy_g1_return, queue_policy_g2_return = PerformRollout(env, queue_policy, config_args)

        print(f"    > Speed overage policy system g1 return: {sum(threshold_policy_g1_return.values())}")
        print(f"    > Speed overage policy system g2 return: {sum(threshold_policy_g2_return.values())}")
        print(f"    > Queue policy system g1 return: {sum(queue_policy_g1_return.values())}")
        print(f"    > Queue policy system g2 return: {sum(queue_policy_g2_return.values())}")

        # Log the online rollout results
        with open(f"{csv_save_dir}/online_rollouts.csv", "a", newline="") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=['round'] +
                                                        [agent + '_mean_policy_g1_return' for agent in agents] +
                                                        ['mean_policy_system_return_g1'] +
                                                        [agent + '_mean_policy_g2_return' for agent in agents] +
                                                        ['mean_policy_system_return_g2'] +

                                                        [agent + '_current_policy_g1_return' for agent in agents] +
                                                        ['current_policy_system_return_g1'] +
                                                        [agent + '_current_policy_g2_return' for agent in agents] +
                                                        ['current_policy_system_return_g2'] +

                                                        [agent + '_threshold_policy_g1_return' for agent in agents] +
                                                        ['threshold_policy_system_return_g1'] +
                                                        [agent + '_threshold_policy_g2_return' for agent in agents] +
                                                        ['threshold_policy_system_return_g2'] +

                                                        [agent + '_queue_policy_g1_return' for agent in agents] +
                                                        ['queue_policy_system_return_g1'] +
                                                        [agent + '_queue_policy_g2_return' for agent in agents] +
                                                        ['queue_policy_system_return_g2'])
            new_row = {}
            new_row['round'] = t
            for agent in agents:
                new_row[agent + '_mean_policy_g1_return'] = mean_policy_g1_return[agent]
                new_row[agent + '_mean_policy_g2_return'] = mean_policy_g2_return[agent]

                new_row[agent + '_current_policy_g1_return'] = current_policy_g1_return[agent]
                new_row[agent + '_current_policy_g2_return'] = current_policy_g2_return[agent]

                new_row[agent + '_threshold_policy_g1_return'] = threshold_policy_g1_return[agent]
                new_row[agent + '_threshold_policy_g2_return'] = threshold_policy_g2_return[agent]

                new_row[agent + '_queue_policy_g1_return'] = queue_policy_g1_return[agent]
                new_row[agent + '_queue_policy_g2_return'] = queue_policy_g2_return[agent]


            new_row['mean_policy_system_return_g1'] = sum(mean_policy_g1_return.values())
            new_row['mean_policy_system_return_g2'] = sum(mean_policy_g2_return.values())

            new_row['current_policy_system_return_g1'] = sum(current_policy_g1_return.values())
            new_row['current_policy_system_return_g2'] = sum(current_policy_g2_return.values())

            new_row['threshold_policy_system_return_g1'] = sum(threshold_policy_g1_return.values())
            new_row['threshold_policy_system_return_g2'] = sum(threshold_policy_g2_return.values())

            new_row['queue_policy_system_return_g1'] = sum(queue_policy_g1_return.values())
            new_row['queue_policy_system_return_g2'] = sum(queue_policy_g2_return.values())

            csv_writer.writerow({**new_row})


        # Capture the round execution time
        round_completeion_time = datetime.now()
        print(f" >> Round {t} of {max_num_rounds} complete!")
        print(f" >> Round execution time: {round_completeion_time-round_start_time}")

    function_stop_time = datetime.now()
    print(f" > Batch offline reinforcement learning complete")
    print(f" > Total execution time: {function_stop_time-function_start_time}")

    lambdas = [lambda_1, lambda_2]
    mean_lambdas = [mean_lambda_1, mean_lambda_2]

    return mean_policies, policies, mean_lambdas, lambdas


def FittedQIteration(observation_spaces:dict,
                     action_spaces:dict,
                     agents:list,
                     dataset:dict,
                     csv_save_dir:str,
                     config_args,
                     constraint:str="",
                     lambda_1:float=None,
                     lambda_2:float=None) -> dict:
    """
    Implementation of Fitted Q Iteration with function approximation for offline learning of a policy
    Note that this implementation utilizes an "actor-critic" approach to solve the RL problem
    (algorithm 4 from Le, et. al)

    :param observation_spaces: Dictionary that maps an agent to the dimensions of the env's raw observations space (assumes no one hot 
            encoding has been applied)
    :param action_spaces: Dictionary that maps agent to its action space
    :param agents: List of agent names
    :param dataset: Dictionary that maps each agent to its experience tuple
    :param csv_save_dir: Path to the directory being used to store CSV files for this experiment
    :param config_args: Configuration arguments used to set up the experiment
    :param constraint: 'speed_overage', 'queue', or 'weighted_sum' defines how the target should be determined while learning the policy
    :param lambda_1: Weight corresponidng to the G1 constraints, if constraint is 'weighted_sum' this must be provided
    :param lambda_2: Weight corresponidng to the G2 constraints, if constraint is 'weighted_sum' this must be provided
    :returns A dictionary that maps each agent to its learned policy
    """
    
    if config_args.global_obs:
        print("ERROR: global observations not supported for FittedQIteration")
        return {}

    print(" >> Beginning Fitted Q Iteration")
    start_time = datetime.now()

    if config_args.parameter_sharing_model:
        # Parameter sharing is used so there is only a single actor and critic network
        print(f"  >>> Parameter sharing enabled")
        eg_agent = agents[0]
        num_agents = len(agents)

        # TODO: change size of observation space if global observations are being used
        observation_space_shape = np.array(observation_spaces[eg_agent].shape).prod() + num_agents  # Convert (X,) shape from tuple to int so it can be modified
        observation_space_shape = tuple(np.array([observation_space_shape]))   
            
        q_network = QNetwork(observation_space_shape, action_spaces[eg_agent].n).to(device) 
        actor_network = Actor(observation_space_shape, action_spaces[eg_agent].n).to(device)
        target_network = QNetwork(observation_space_shape, action_spaces[eg_agent].n).to(device)
        target_network.load_state_dict(q_network.state_dict())    # Intialize the target network the same as the main network
        optimizer = optim.Adam(q_network.parameters(), lr=config_args.learning_rate) # All agents use the same optimizer for training
        actor_optimizer = optim.Adam(list(actor_network.parameters()), lr=config_args.learning_rate)
        actor_losses = {agent: None for agent in agents}       # Dictionary that maps each agent to the loss values for its actor network (used for logging)

    else:
        print(f"  >>> Parameter sharing NOT enabled")
        q_network = {}          # Dictionary for storing q-networks (maps agent to a q-network)
        actor_network = {}      # Dictionary for storing actor networks (maps agents to a network)
        target_network = {}     # Dictionary for storing target networks (maps agent to a network)
        optimizer = {}          # Dictionary for storing optimizers for each RL problem
        actor_optimizer = {}    # Dictionary for storing the optimizers used to train the actor networks 
        actor_losses = {}       # Dictionary that maps each agent to the loss values for its actor network (used for logging)

        for agent in agents:
            observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if config_args.global_obs else observation_spaces[agent].shape
            q_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device) 
            actor_network[agent] = Actor(observation_space_shape, action_spaces[agent].n).to(device)
            target_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
            target_network[agent].load_state_dict(q_network[agent].state_dict())    # Intialize the target network the same as the main network
            optimizer[agent] = optim.Adam(q_network[agent].parameters(), lr=config_args.learning_rate) # All agents use the same optimizer for training
            actor_optimizer[agent] = optim.Adam(list(actor_network[agent].parameters()), lr=config_args.learning_rate)
            actor_losses[agent] = None

    # Define loss functions for the critic and actor
    loss_fn = nn.MSELoss()
    actor_loss_fn = nn.CrossEntropyLoss()

    # Define a single experience sample that can be used to periodically evaluate the network to test for convergence
    # NOTE: Observations sampled from the dataset should already have one-hot encoding applied to them for parameter sharing
    eval_obses, eval_actions, eval_next_obses, eval_g1s, eval_g2s, eval_dones = dataset[agents[0]].sample(1)

    for global_step in range(config_args.total_timesteps):

        # TODO: remove?
        if (global_step % config_args.train_frequency == 0):  
                
            if config_args.parameter_sharing_model:
                # Agent is randomly selected to be used for calculating the Q values and targets
                random_agent = random.choice(agents)

                # Sample data from the dataset
                # NOTE: Observations sampled from the dataset should already have one-hot encoding applied to them for parameter sharing
                s_obses, s_actions, s_next_obses, s_g1s, s_g2s, s_dones = dataset[random_agent].sample(config_args.batch_size)
                # print(f"s_obses {s_obses}, s_actions {s_actions}, s_next_obses {s_next_obses}, s_g1s {s_g1s}, s_g2s {s_g2s}, s_dones {s_dones}")
                # Compute the target
                with torch.no_grad():
                    # Calculate max_a Q(s',a)
                    # NOTE: that the original FQI in the paper used min_a here but our constaint values are the negative version of theirs
                    # so we need to take max here
                    target_max = torch.max(target_network.forward(s_next_obses), dim=1).values
                    
                    # Calculate the full TD target 
                    # NOTE: that the target in this Fitted Q iteration implementation depends on the type of constraint we are using to 
                    # learn the policy
                    if (constraint == "speed_overage"):
                        # Use the "g1" constraint
                        td_target = torch.Tensor(s_g1s).to(device) + config_args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))

                    elif (constraint == "queue"):
                        # Use the "g2" constraint
                        td_target = torch.Tensor(s_g2s).to(device) + config_args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))

                    elif (constraint == "weighted_sum"):
                        # Use both g1 and g2 but weighted with lambda
                        td_target = torch.Tensor(lambda_1 * s_g1s).to(device) + torch.Tensor(lambda_2 * s_g2s).to(device) + config_args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))
                    else: 
                        print(f"ERROR: Constraint function '{constraint}' not recognized, unable to train using Fitted Q Iteration")
                        return {}

                q_values = q_network.forward(s_obses)
                old_val = q_network.forward(s_obses).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
                loss = loss_fn(td_target, old_val)

                # Optimize the model for the critic
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(q_network.parameters()), config_args.max_grad_norm)
                optimizer.step()


                # Actor training
                a, log_pi, action_probs = actor_network.to(device).get_action(s_obses)

                # Compute the loss for this agent's actor
                # NOTE: Actor uses cross-entropy loss function where
                # input is the policy dist and the target is the value function with one-hot encoding applied
                # Q-values from "critic" encoded so that the highest state-action value maps to a probability of 1
                q_values_one_hot = one_hot_q_values(q_values)
                actor_loss = actor_loss_fn(action_probs, q_values_one_hot.to(device))
                # NOTE: the actor losses are only updated for the random agent here, not all agents so at this time step only the random agent
                # has current values for loss, this should not matter for showing trends in the loss function through
                actor_losses[random_agent] = actor_loss.item() 

                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(list(actor_network.parameters()), config_args.max_grad_norm)
                actor_optimizer.step()

                # Update the target network
                if (global_step % config_args.target_network_frequency == 0):
                    target_network.load_state_dict(q_network.state_dict())

                
                # Periodically log data to CSV
                if (global_step % 1000 == 0):

                    # Evaluate the probability of the first agent selecting the first action from the evaluation state
                    a, log_pi, action_probs = actor_network.get_action(eval_obses)
                    first_agent_first_action_probs = (action_probs.squeeze())[0].item() # Cast from tensor object to float

                    with open(f"{csv_save_dir}/FQI_actor_loss.csv", "a", newline="") as csvfile:
                        csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['Pi(a|s) Sample', 'global_step'])
                        csv_writer.writerow({**actor_losses, **{'Pi(a|s) Sample' : first_agent_first_action_probs,
                                                                    'global_step' : global_step}})

            else:

                # No parameter sharing so each agent is trained individually
                for agent in agents:

                    # Sample data from the dataset
                    s_obses, s_actions, s_next_obses, s_g1s, s_g2s, s_dones = dataset[agent].sample(config_args.batch_size)
                    # print(f"s_obses {s_obses}, s_actions {s_actions}, s_next_obses {s_next_obses}, s_g1s {s_g1s}, s_g2s {s_g2s}, s_dones {s_dones}")
                    # Compute the target
                    with torch.no_grad():
                        # Calculate max_a Q(s',a)
                        # NOTE: that the original FQI in the paper used min_a here but our constaint values are the negative version of theirs
                        # so we need to take max here
                        target_max = torch.max(target_network[agent].forward(s_next_obses), dim=1).values
                        
                        # Calculate the full TD target 
                        # NOTE: that the target in this Fitted Q iteration implementation depends on the type of constraint we are using to 
                        # learn the policy
                        if (constraint == "speed_overage"):
                            # Use the "g1" constraint
                            td_target = torch.Tensor(s_g1s).to(device) + config_args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))

                        elif (constraint == "queue"):
                            # Use the "g2" constraint
                            td_target = torch.Tensor(s_g2s).to(device) + config_args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))

                        elif (constraint == "weighted_sum"):
                            # Use both g1 and g2 but weighted with lambda
                            td_target = torch.Tensor(lambda_1 * s_g1s).to(device) + torch.Tensor(lambda_2 * s_g2s).to(device) + config_args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))
                        else: 
                            print(f"ERROR: Constraint function '{constraint}' not recognized, unable to train using Fitted Q Iteration")
                            return {}

                    q_values = q_network[agent].forward(s_obses)
                    old_val = q_network[agent].forward(s_obses).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
                    loss = loss_fn(td_target, old_val)

                    # Optimize the model for the critic
                    optimizer[agent].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(q_network[agent].parameters()), config_args.max_grad_norm)
                    optimizer[agent].step()


                    # Actor training
                    a, log_pi, action_probs = actor_network[agent].to(device).get_action(s_obses)

                    # Compute the loss for this agent's actor
                    # NOTE: Actor uses cross-entropy loss function where
                    # input is the policy dist and the target is the value function with one-hot encoding applied
                    # Q-values from "critic" encoded so that the highest state-action value maps to a probability of 1
                    q_values_one_hot = one_hot_q_values(q_values)
                    actor_loss = actor_loss_fn(action_probs, q_values_one_hot.to(device))
                    actor_losses[agent] = actor_loss.item()

                    actor_optimizer[agent].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(list(actor_network[agent].parameters()), config_args.max_grad_norm)
                    actor_optimizer[agent].step()

                    # Update the target network
                    if global_step % config_args.target_network_frequency == 0:
                        target_network[agent].load_state_dict(q_network[agent].state_dict())

                    
                # Periodically log data to CSV
                if (global_step % 1000 == 0):

                    # Evaluate the probability of the first agent selecting the first action from the evaluation state
                    a, log_pi, action_probs = actor_network[agents[0]].get_action(eval_obses)
                    first_agent_first_action_probs = (action_probs.squeeze())[0].item() # Cast from tensor object to float

                    with open(f"{csv_save_dir}/FQI_actor_loss.csv", "a", newline="") as csvfile:
                        csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['Pi(a|s) Sample', 'global_step'])
                        csv_writer.writerow({**actor_losses, **{'Pi(a|s) Sample' : first_agent_first_action_probs,
                                                                    'global_step' : global_step}})

    stop_time = datetime.now()
    print(" >> Fitted Q Iteration complete")
    print(" >>> Total execution time: {}".format(stop_time-start_time))
    
    # If we're using parameter sharing, there is only a single network so to conform to the rest of the code, return a dictionary that maps
    # each agent to it's policy network
    if config_args.parameter_sharing_model:
        actor_network = {agent: actor_network for agent in agents}

    return actor_network


def FittedQEvaluation(observation_spaces:dict,
                     action_spaces:dict,
                     agents:list,
                     policy:dict,
                     dataset:dict,
                     csv_save_dir:str,
                     csv_file_suffix:str,
                     config_args,
                     constraint:str="") -> dict:
    """
    Implementation of Fitted Off-Policy Evaluation with function approximation for offline evaluation
    of a policy according to a provided constraint
    (algorithm 3 from Le, et. al)

    :param observation_spaces: Dictionary that maps an agent to the observation space dimensions
    :param action_spaces: Dictionary that maps agent to its action space
    :param agents: List of agent names
    :param policy: Dictionary that maps an agent to its policy to be evaluated
    :param dataset: Dictionary that maps each agent to its experience tuple
    :param csv_save_dir: Path to the directory being used to store CSV files for this experiment
    :param csv_file_suffix: String to append to the name of the csv file to differentiate between which mean constraint value function is being evaluated
    :param config_args: Configuration arguments used to set up the experiment
    :param constraint: 'speed_overage' or 'queue', defines how the target should be determined while learning the value function
    :returns A dictionary that maps each agent to its learned constraint value function
    """

    # TODO: implement
    if config_args.global_obs:
        print("ERROR: global observations not supported for FittedQEvaluation")
        return {}

    print(" >> Beginning Fitted Q Evaluation")
    start_time = datetime.now()


    if config_args.parameter_sharing_model:
        # Parameter sharing is used so there is only a single actor and critic network
        print(f"  >>> Parameter sharing enabled")
        eg_agent = agents[0]
        onehot_keys = {agent: i for i, agent in enumerate(agents)}
        num_agents = len(agents)
        for agent in agents:
            onehot = np.zeros(num_agents)
            onehot[onehot_keys[agent]] = 1.0

        # TODO: change size of observation space if global observations are being used
        observation_space_shape = np.array(observation_spaces[eg_agent].shape).prod() + num_agents  # Convert (X,) shape from tuple to int so it can be modified
        observation_space_shape = tuple(np.array([observation_space_shape]))   
        
            
        q_network = QNetwork(observation_space_shape, action_spaces[eg_agent].n).to(device) 
        target_network = QNetwork(observation_space_shape, action_spaces[eg_agent].n).to(device)
        target_network.load_state_dict(q_network.state_dict())    # Intialize the target network the same as the main network
        optimizer = optim.Adam(q_network.parameters(), lr=config_args.learning_rate) # All agents use the same optimizer for training
        losses = {agent: None for agent in agents}         # Dictionary that maps each agent to the loss values for the critic network


    else:
        print(f"  >>> Parameter sharing NOT enabled")

        q_network = {}      # Dictionary for storing q-networks (maps agent to a q-network)
        target_network = {} # Dictionary for storing target networks (maps agent to a network)
        optimizer = {}      # Dictionary for storing optimizers for each RL problem
        actions = {}        # Dictionary that maps each agent to the action it selected
        losses = {}         # Dictionary that maps each agent to the loss values for its critic network

        for agent in agents:
            observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if config_args.global_obs else observation_spaces[agent].shape
            q_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
            target_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
            target_network[agent].load_state_dict(q_network[agent].state_dict())    # Intialize the target network the same as the main network
            optimizer[agent] = optim.Adam(q_network[agent].parameters(), lr=config_args.learning_rate) # All agents use the same optimizer for training
            actions[agent] = None

    # Define loss function as MSE loss
    loss_fn = nn.MSELoss()

    # Define a single experience sample that can be used to periodically evaluate the network to test for convergence
    # NOTE: Observations sampled from the dataset should already have one-hot encoding applied to them for parameter sharing
    eval_obses, eval_actions, eval_next_obses, eval_g1s, eval_g2s, eval_dones = dataset[agents[0]].sample(1)

    # TODO: this should be updated to be for k = 1:K (does not need to be the same as total_timesteps)
    for global_step in range(config_args.total_timesteps):

        if (global_step % config_args.train_frequency == 0):  


            if config_args.parameter_sharing_model:
                # Agent is randomly selected to be used for calculating the Q values and targets
                random_agent = random.choice(agents)

                # Sample data from the dataset
                # NOTE: Observations sampled from the dataset should already have one-hot encoding applied to them for parameter sharing
                s_obses, s_actions, s_next_obses, s_g1s, s_g2s, s_dones = dataset[random_agent].sample(config_args.batch_size)

                # Use the sampled next observations (x') to generate actions according to the provided policy
                # NOTE this method of getting actions is identical to how it is performed in actor-critic
                actions_for_agent, _, _ = policy[random_agent].get_action(s_obses)

                # Compute the target
                # NOTE That this is the only thing different between FQE and FQI
                with torch.no_grad():
                    
                    # Calculate Q(s',pi(s'))
                    target = target_network.forward(s_next_obses).gather(1, torch.Tensor(actions_for_agent).view(-1,1).to(device)).squeeze()
                    
                    # Calculate the full TD target 
                    # NOTE that the target in this Fitted Q iteration implementation depends on the type of constraint we are using to 
                    # learn the policy
                    if (constraint == "speed_overage"):
                        # Use the "g1" constraint
                        td_target = torch.Tensor(s_g1s).to(device) + config_args.gamma * target * (1 - torch.Tensor(s_dones).to(device))

                    elif (constraint == "queue"):
                        # Use the "g2" constraint
                        td_target = torch.Tensor(s_g2s).to(device) + config_args.gamma * target * (1 - torch.Tensor(s_dones).to(device))

                    else: 
                        print("ERROR: Constraint function '{}' not recognized, unable to train using Fitted Q Iteration".format(constraint))

                # Calculate "previous" value using actions from the experience dataset
                old_val = q_network.forward(s_obses).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()

                loss = loss_fn(td_target, old_val)
                losses[random_agent] = loss.item()

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(q_network.parameters()), config_args.max_grad_norm)
                optimizer.step()

                # Update the target network
                if (global_step % config_args.target_network_frequency == 0):
                    target_network.load_state_dict(q_network.state_dict())

                # Periodically log data to CSV
                if (global_step % 1000 == 0):

                    # Evaluate the Q(s,a) of the first agent selecting the first action from the evaluation state
                    eval_q_s_a = q_network.forward(eval_obses).squeeze()
                    first_agent_first_action_value = eval_q_s_a[0].item()   # cast the tensor object to a float

                    with open(f"{csv_save_dir}/FQE_loss_{csv_file_suffix}.csv", "a", newline="") as csvfile:
                        csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['Q(s,a) Sample', 'global_step'])
                        csv_writer.writerow({**losses, **{'Q(s,a) Sample' : first_agent_first_action_value,
                                                                    'global_step' : global_step}})

            else:
                # Training for each agent
                for agent in agents:
                
                    
                    # Sample data from the dataset
                    s_obses, s_actions, s_next_obses, s_g1s, s_g2s, s_dones = dataset[agent].sample(config_args.batch_size)
                    
                    # Use the sampled next observations (x') to generate actions according to the provided policy
                    # NOTE this method of getting actions is identical to how it is performed in actor-critic
                    actions_for_agent, _, _ = policy[agent].get_action(s_obses)
                    
                    # Compute the target
                    # NOTE That this is the only thing different between FQE and FQI
                    with torch.no_grad():
                        
                        # Calculate Q(s',pi(s'))
                        target = target_network[agent].forward(s_next_obses).gather(1, torch.Tensor(actions_for_agent).view(-1,1).to(device)).squeeze()
                        
                        # Calculate the full TD target 
                        # NOTE that the target in this Fitted Q iteration implementation depends on the type of constraint we are using to 
                        # learn the policy
                        if (constraint == "speed_overage"):
                            # Use the "g1" constraint
                            td_target = torch.Tensor(s_g1s).to(device) + config_args.gamma * target * (1 - torch.Tensor(s_dones).to(device))

                        elif (constraint == "queue"):
                            # Use the "g2" constraint
                            td_target = torch.Tensor(s_g2s).to(device) + config_args.gamma * target * (1 - torch.Tensor(s_dones).to(device))

                        else: 
                            print("ERROR: Constraint function '{}' not recognized, unable to train using Fitted Q Iteration".format(constraint))

                    # Calculate "previous" value using actions from the experience dataset
                    old_val = q_network[agent].forward(s_obses).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()

                    loss = loss_fn(td_target, old_val)
                    losses[agent] = loss.item()

                    # optimize the model
                    optimizer[agent].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(q_network[agent].parameters()), config_args.max_grad_norm)
                    optimizer[agent].step()

                    # Update the target network
                    if global_step % args.target_network_frequency == 0:
                        target_network[agent].load_state_dict(q_network[agent].state_dict())


                # Periodically log data to CSV
                if (global_step % 1000 == 0):

                    # Evaluate the Q(s,a) of the first agent selecting the first action from the evaluation state
                    eval_q_s_a = q_network[agents[0]].forward(eval_obses).squeeze()
                    first_agent_first_action_value = eval_q_s_a[0].item()   # cast the tensor object to a float

                    with open(f"{csv_save_dir}/FQE_loss_{csv_file_suffix}.csv", "a", newline="") as csvfile:
                        csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['Q(s,a) Sample', 'global_step'])
                        csv_writer.writerow({**losses, **{'Q(s,a) Sample' : first_agent_first_action_value,
                                                                    'global_step' : global_step}})

    stop_time = datetime.now()
    print(" >> Fitted Q Evaluation complete")
    print(" >>> Total execution time: {}".format(stop_time-start_time))

    # If we're using parameter sharing, there is only a single network so to conform to the rest of the code, return a dictionary that maps
    # each agent to it's constraint network
    if config_args.parameter_sharing_model:
        q_network = {agent: q_network for agent in agents}

    return q_network


def CalculateMeanPolicy(latest_learned_policy:dict,
                        previous_mean_policy:dict,
                        observation_spaces:dict,
                        action_spaces:dict,
                        agents:list,
                        round:int,
                        dataset:dict,
                        csv_save_dir:str,
                        config_args) -> dict:
    """
    Calculate the "mean" policy using the the previous mean policy and the last learned policy

    :param latest_learned_policy: Dictionary of policies that was just learned during this round
    :param previous_mean_policy: The previously learned mean policy (i.e. the output of this function from the last round)
    :param observation_spaces: Dictionary that maps agents to observations spaces
    :param action_spaces: Dictionary that maps agents to action spaces
    :param agents: List of agents in the environmnet
    :param round: The current round of batch offline RL being performed
    :param dataset: Dictionary that maps agents to a collection of experience tuples
    :param csv_save_dir: Path to the directory being used to store CSV files for this experiment
    :param config_args: Config arguments used to set up the experiment
    :returns a Dictionary that maps each agent to its expectation E_t[pi] = 1/t*(pi_t + (t-1)*E_t-1[pi])
    """

    print(f" >> Evaluating Mean Policy")
    start_time = datetime.now()

    losses = {agent: None for agent in agents}       # Dictionary that maps each agent to the loss values for its network
    num_agents = len(agents)
    eg_agent = agents[0]    

    if config_args.parameter_sharing_model:
        # Create a single network for the mean policy
        if config_args.global_obs:
            observation_space_shape = tuple((shape+1) * (num_agents) for shape in observation_spaces[eg_agent].shape)

        else:
            observation_space_shape = np.array(observation_spaces[eg_agent].shape).prod() + num_agents

        mean_policy = Actor(observation_space_shape, action_spaces[eg_agent].n).to(device)
        optimizer = optim.Adam(mean_policy.parameters(), lr=config_args.learning_rate)

    else:
        # Create a separate policy network for each agent
        mean_policy = {}    # Dictionary that maps agents to the "mean" policy for this round
        optimizer = {}      # Dictionary for storing optimizer for each agent's network
        
        for agent in agents: 
            observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if config_args.global_obs else observation_spaces[agent].shape

            mean_policy[agent] = Actor(observation_space_shape, action_spaces[agent].n).to(device)

            # Initialize the mean policy using the previous one
            mean_policy[agent].load_state_dict(previous_mean_policy[agent].state_dict())

            optimizer[agent] = optim.Adam(mean_policy[agent].parameters(), lr=config_args.learning_rate) # All agents use the same optimizer for training

    loss_fn = nn.MSELoss() # TODO: should this be MSE or Cross Entropy?

    # For k = 1:K
    for global_step in range(config_args.total_timesteps):

        if (global_step % config_args.train_frequency == 0):  

            if config_args.parameter_sharing_model:
                # Agent is randomly selected to be used for calculating the Q values and targets
                random_agent = random.choice(agents)

                # Sample data from the dataset
                # NOTE: when using parameter sharing, the observations here should already have 
                # one hot encoding applied
                s_obses, s_actions, s_next_obses, s_g1s, s_g2s, s_dones = dataset[random_agent].sample(config_args.batch_size)

                with torch.no_grad():
                    # NOTE: when using parameter sharing, the latest_learned_policy dict should contain the same network
                    # for each agent (same for the previous_mean_policy)

                    # Get the action probability distribution from the latest learned policy
                    _, _, latest_learned_policy_probs = latest_learned_policy[random_agent].get_action(s_obses)
                    
                    # Get the action probability distribution from the last mean policy
                    _, _, prev_mean_policy_probs = previous_mean_policy[random_agent].get_action(s_obses)

                    # Compute the target
                    target =  1/round*(latest_learned_policy_probs + (round - 1.0) * prev_mean_policy_probs)

                # Get the action probability distribution from the previous state of the mean policy
                _, _, old_policy_probs = mean_policy.get_action(s_obses)

                # Calculate the loss between the "old" mean policy and the target
                loss = loss_fn(target, old_policy_probs)     # TODO: discuss loss function with chihui
                losses[random_agent] = loss.item()

                # Optimize the model 
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(mean_policy.parameters()), config_args.max_grad_norm)
                optimizer.step()

            else:

                # No parameter sharing so need to train each agent
                for agent in agents:
                    
                    # Sample data from the dataset
                    s_obses, s_actions, s_next_obses, s_g1s, s_g2s, s_dones = dataset[agent].sample(config_args.batch_size)

                    with torch.no_grad():
                        
                        # Get the action probability distribution from the latest learned policy
                        _, _, latest_learned_policy_probs = latest_learned_policy[agent].get_action(s_obses)
                        
                        # Get the action probability distribution from the last mean policy
                        _, _, prev_mean_policy_probs = previous_mean_policy[agent].get_action(s_obses)

                        # Compute the target
                        target =  1/round*(latest_learned_policy_probs + (round - 1.0) * prev_mean_policy_probs)

                    # Get the action probability distribution from the previous state of the mean policy
                    _, _, old_policy_probs = mean_policy[agent].get_action(s_obses)

                    # Calculate the loss between the "old" mean policy and the target
                    loss = loss_fn(target, old_policy_probs)     # TODO: discuss loss function with chihui
                    losses[agent] = loss.item()

                    # Optimize the model 
                    optimizer[agent].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(mean_policy[agent].parameters()), config_args.max_grad_norm)
                    optimizer[agent].step()

        # Periodically log data to CSV
        if (global_step % 1000 == 0):

            with open(f"{csv_save_dir}/mean_policy_loss.csv", "a", newline="") as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['global_step', 'round'])
                csv_writer.writerow({**losses, **{'global_step' : global_step, 'round' : round}})

    stop_time = datetime.now()
    print(" >> Mean policy evaluation complete")
    print(" >>> Total execution time: {}".format(stop_time-start_time))
    
    # If we're using parameter sharing, there is only a single network so to conform to the rest of the code, return a dictionary that maps
    # each agent to it's policy network
    if config_args.parameter_sharing_model:
        mean_policy = {agent: mean_policy for agent in agents}

    return mean_policy


def CalculateMeanConstraint(latest_learned_constraint:dict,
                            previous_mean_constraint:dict,
                            observation_spaces:dict,
                            action_spaces:dict,
                            agents:list,
                            round:int,
                            dataset:dict,
                            csv_save_dir:str,
                            csv_file_suffix:str,
                            config_args) -> dict:
    """
    Calculate the "mean" constraint value function using the previous mean constraint value function and the last
    learned constraint value function

    :param latest_learned_constraint: Dictionary of constraint value functions that was just learned during this round
    :param previous_mean_constraint: The previously learned constraint value function (i.e. the output of this function from the last round)
    :param observation_spaces: Dictionary that maps agents to observations spaces
    :param action_spaces: Dictionary that maps agents to action spaces
    :param agents: List of agents in the environmnet
    :param round: The current round of batch offline RL being performed
    :param dataset: Dictionary that maps agents to a collection of experience tuples
    :param csv_save_dir: Path to the directory being used to store CSV files for this experiment
    :param csv_file_suffix: String to append to the name of the csv file to differentiate between which mean constraint value function is being evaluated
    :param config_args: Config arguments used to set up the experiment
    :returns a Dictionary that maps each agent to its expectation E_t[g] = 1/t*(g_t + (t-1)*E_t-1[g])
    """
    print(f" >> Evaluating Mean Constraint: '{csv_file_suffix}'")
    start_time = datetime.now()

    losses = {agent: None for agent in agents}       # Dictionary that maps each agent to the loss values for its network
    num_agents = len(agents)
    eg_agent = agents[0]    

    if config_args.parameter_sharing_model:
        # Create a single network for the mean policy
        if config_args.global_obs:
            observation_space_shape = tuple((shape+1) * (num_agents) for shape in observation_spaces[eg_agent].shape)

        else:
            observation_space_shape = np.array(observation_spaces[eg_agent].shape).prod() + num_agents

        mean_constraint = QNetwork(observation_space_shape, action_spaces[eg_agent].n).to(device)
        optimizer = optim.Adam(mean_constraint.parameters(), lr=config_args.learning_rate)

    else:

        mean_constraint = {}    # Dictionary that maps agents to the "mean" constraint value function for this round
        optimizer = {}          # Dictionary for storing optimizer for each agent's network

        for agent in agents: 
            observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if config_args.global_obs else observation_spaces[agent].shape


            mean_constraint[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)

            # Initialize the mean policy using the previous one
            mean_constraint[agent].load_state_dict(previous_mean_constraint[agent].state_dict())

            optimizer[agent] = optim.Adam(mean_constraint[agent].parameters(), lr=config_args.learning_rate) # All agents use the same optimizer for training
            losses[agent] = None

    loss_fn = nn.MSELoss() # TODO: should this be MSE or Cross Entropy?

    # For k = 1:K
    for global_step in range(config_args.total_timesteps):
        
        # TODO: remove?
        if (global_step % config_args.train_frequency == 0):  
            
            if config_args.parameter_sharing_model:
                # Agent is randomly selected to be used for calculating the Q values and targets
                random_agent = random.choice(agents)

                # Sample data from the dataset
                # NOTE: when using parameter sharing, the observations here should already have 
                # one hot encoding applied
                s_obses, s_actions, s_next_obses, s_g1s, s_g2s, s_dones = dataset[random_agent].sample(config_args.batch_size)

                with torch.no_grad():
                    # NOTE: when using parameter sharing, the latest_learned_constraint dict should contain the same network
                    # for each agent (same for the previous_mean_constraint)
                                        
                    # Get the action probability distribution from the latest learned value function
                    q_values_latest = latest_learned_constraint[random_agent].forward(s_obses).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
                    
                    # Get the action probability distribution from the last mean value function
                    q_values_prev_mean = previous_mean_constraint[random_agent].forward(s_obses).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()

                    # Compute the target
                    target =  1/round*(q_values_latest + (round - 1.0) * q_values_prev_mean)

                # Get the action probability distribution from the previous state of the mean policy
                old_val = mean_constraint.forward(s_obses).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()

                # Compute loss 
                loss = loss_fn(target, old_val)
                losses[random_agent] = loss.item()

                # Optimize the model 
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(mean_constraint.parameters()), config_args.max_grad_norm)
                optimizer.step()

            else:

                # Training for each agent
                for agent in agents:
                    
                    # Sample data from the dataset
                    s_obses, s_actions, s_next_obses, s_g1s, s_g2s, s_dones = dataset[agent].sample(config_args.batch_size)

                    with torch.no_grad():
                        
                        # Get the action probability distribution from the latest learned value function
                        q_values_latest = latest_learned_constraint[agent].forward(s_obses).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
                        
                        # Get the action probability distribution from the last mean value function
                        q_values_prev_mean = previous_mean_constraint[agent].forward(s_obses).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()

                        # Compute the target
                        target =  1/round*(q_values_latest + (round - 1.0) * q_values_prev_mean)

                    # Get the action probability distribution from the previous state of the mean policy
                    old_val = mean_constraint[agent].forward(s_obses).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()

                    # Compute loss 
                    loss = loss_fn(target, old_val)
                    losses[agent] = loss.item()

                    # Optimize the model 
                    optimizer[agent].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(mean_constraint[agent].parameters()), config_args.max_grad_norm)
                    optimizer[agent].step()

        # Periodically log data to CSV
        if (global_step % 1000 == 0):

            with open(f"{csv_save_dir}/mean_constraint_loss_{csv_file_suffix}.csv", "a", newline="") as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['global_step', 'round'])
                csv_writer.writerow({**losses, **{'global_step' : global_step, 'round' : round}})

    stop_time = datetime.now()
    print(" >> Mean constraint function evaluation complete")
    print(" >>> Total execution time: {}".format(stop_time-start_time))

    # If we're using parameter sharing, there is only a single network so to conform to the rest of the code, return a dictionary that maps
    # each agent to it's policy network
    if config_args.parameter_sharing_model:
        mean_constraint = {agent: mean_constraint for agent in agents}

    return mean_constraint


def PerformRollout(env:sumo_rl.parallel_env, policy:dict, config_args)->tuple[dict, dict, dict]:
    """
    Perform a 1-episode rollout of a provided policy to evaluate the constraint functions g1 and g2.
    This function assumes that the environment has been set up with the 'queue' reward function when evaluating the
    g1 and g2 constraints.

    NOTE: this is an "online" rollout because it assumes that all agents are using their current learned policy

    :param env: The environment to execute the policy in
    :param policy: Dictionary that maps agents to "actor" models
    :param config_args: Configuration arguments used to set up the experiment
    :returns: Three dictionaries, the first dict maps agents to their accumulated reward for the episode,
            the second dict maps agents to their accumulated g1 constraint for the episode, the third dict
            maps agents to their accumulated g2 constraint for the episode
    """
    # TODO: update function to support global observations

    # Define the speed overage threshold used to evaluate the g1 constraint
    # (note this needs to match what is used in GenerateDataset)
    SPEED_OVERAGE_THRESHOLD = 13.89
    # SPEED_LOWER_THRESHOLD = 0.01
    # SPEED_LOWER_THRESHOLD = 5.0
    # SPEED_LOWER_THRESHOLD = 0.0
    SPEED_LOWER_THRESHOLD = 1.0

    agents = env.possible_agents

    # This function assumes that the environment is set up with the 'queue' reward so if that's not the case 
    # just remind the user
    if (config_args.sumo_reward != 'queue'):
        print(f"  > WARNING: Reward '{config_args.sumo_reward}' specified but being ignored\n" \
              f"    > Online rollout being performed with 'queue' reward.") 

    # Define empty dictionary that maps agents to actions
    actions = {agent: None for agent in agents}

    # Dictionary that maps the each agent to its cumulative reward each episode
    episode_rewards = {agent: 0.0 for agent in agents}

    # Maps each agent to its MAX SPEED OVERAGE for this step
    episode_constraint_1 = {agent : 0.0 for agent in agents}

    # Maps each agent to the accumulated NUBMER OF CARS STOPPED for episode
    episode_constraint_2 = {agent : 0.0 for agent in agents}

    # Initialize the env
    obses, _ = env.reset()

    if config_args.parameter_sharing_model:
        # Apply one-hot encoding to the initial observations
        onehot_keys = {agent: i for i, agent in enumerate(agents)}

        for agent in agents:
            onehot = np.zeros(num_agents)
            onehot[onehot_keys[agent]] = 1.0
            obses[agent] = np.hstack([onehot, obses[agent]])

    # Perform the rollout
    for sumo_step in range(config_args.sumo_seconds):
        # Populate the action dictionary
        for agent in agents:
            # Only use optimal actions according to the policy
            action, _, _ = policy[agent].to(device).get_action(obses[agent])
            actions[agent] = action.detach().cpu().numpy()

        # Apply all actions to the env
        next_obses, rewards, dones, truncated, info = env.step(actions)

        if np.prod(list(dones.values())):
            # DEBUGGING:
            print(f"    > Last max speeds observed by each agent: ")
            for agent in agents:
                print(f"     > agent: {agent}  max speed: {obses[agent][-1]}")
            
            break

        if config_args.parameter_sharing_model:
            # Apply one-hot encoding to the observations
            onehot_keys = {agent: i for i, agent in enumerate(agents)}

            for agent in agents:
                onehot = np.zeros(num_agents)
                onehot[onehot_keys[agent]] = 1.0
                next_obses[agent] = np.hstack([onehot, next_obses[agent]])

        # Accumulate the total episode reward and max speeds
        for agent in agents:
            # At the end of a simulation, next_obses is an empty dictionary so don't log it 
            episode_rewards[agent] += rewards[agent]
            max_speed_observed_by_agent = next_obses[agent][-1]
            episode_constraint_1[agent] += CalculateMaxSpeedOverage(max_speed=max_speed_observed_by_agent, 
                                                                    speed_limit=SPEED_OVERAGE_THRESHOLD,
                                                                    lower_speed_limit=SPEED_LOWER_THRESHOLD)
            episode_constraint_2[agent] += rewards[agent]   # NOTE That right now, the g2 constraint is the same as the 'queue' model

                
        obses = next_obses

    return episode_rewards, episode_constraint_1, episode_constraint_2


def PerformOfflineRollout(value_function:dict, policies:dict, mini_dataset:dict)->dict:
    """
    :param value_function: Dictionary that maps agents to the learned value function for a given constraint
    :param mini_dataset: A subset of the larger dataset that contains experience tuples that are evaluated in the rollout
            using the provided dataset
            NOTE: This function assumes the dataset has been sampled such that mini_dataset contains lists of of experience
            tuples for each agent
    :returns The cummulative return for all agents of the mini dataset according to the provided value functions normalized 
            by the size of the mini dataset
    """

    agents = list(value_function.keys())
    cumulative_return = {agent:0.0 for agent in agents}

    for agent in agents:
        # NOTE: when using parameter sharing, the observations here should already have 
        # one hot encoding applied
        obses_array, _, _, _, _, _ = mini_dataset[agent]

        # Get the max_a Q(s,a) for the observation
        actions_from_policy, _, _ = policies[agent].to(device).get_action(obses_array)

        # Evaluate the actions that were selected by the policies from this observation
        values = (value_function[agent].forward(obses_array)).to(device).gather(1, actions_from_policy.view(-1,1)).squeeze()

        # Add up the max values of all states from the mini dataset
        cumulative_return[agent] = sum(values)/(len(mini_dataset[agent]))
        # print(f"   > cumulative_return for agent '{agent}': {cumulative_return[agent]}")

    return cumulative_return


def OnlineLambdaLearning(lambda_1_prev:float, 
                         lambda_2_prev:float, 
                         g1_returns:dict, 
                         g2_returns:dict)->tuple[float, float]:
    """
    :param lambda_1_prev: The value of lambda1 at the end of the previous round
    :param lambda_2_prev: The value of lambda2 at the end of the previous round
    :param g1_returns: Dictionary that maps agents to the G1 returns from an offline rollout (assessed using the G1 value function for this round)
    :param g2_returns: Dictionary that maps agents to the G2 returns from an offline rollout (assessed using the G1 value function for this round)
    :returns Updated values for lambda1 and lambda2
    """

    # Calculate the accumulated returns (averaged over all agents)
    num_agents = len(g1_returns.keys()) # TODO: add check to make sure g1 and g2 num agents the same
    avg_cumulative_g1_returns = torch.sum(torch.tensor(list(g1_returns.values()))).detach().numpy() / num_agents
    avg_cumulative_g2_returns = torch.sum(torch.tensor(list(g2_returns.values()))).detach().numpy() / num_agents

    # Lambda learning rate # TODO: config
    # n = 0.01
    # n = 0.00001
    n = 0.1

    exp_g1_returns = np.exp(n * avg_cumulative_g1_returns)
    exp_g2_returns = np.exp(n * avg_cumulative_g2_returns)

    lambda_1 = lambda_1_prev * exp_g1_returns / (lambda_1_prev * exp_g1_returns + lambda_2_prev * exp_g2_returns)
    lambda_2 = lambda_2_prev * exp_g2_returns / (lambda_1_prev * exp_g1_returns + lambda_2_prev * exp_g2_returns)

    # Don't let lambda be 0.0
    lambda_1 = max(0.00001, lambda_1)
    lambda_2 = max(0.00001, lambda_2)

    return lambda_1, lambda_2, avg_cumulative_g1_returns, avg_cumulative_g2_returns

def OnlineLambdaLearningByImprovementRate(lambda_1_prev:float,
                            lambda_2_prev:float,
                            g1_returns:dict,
                            g2_returns:dict,
                            avg_cumulative_g1_returns_prev:float,
                            avg_cumulative_g2_returns_prev:float)->(float, float, float, float):
    """
    :param lambda_1_prev: The value of lambda1 at the end of the previous round
    :param lambda_2_prev: The value of lambda2 at the end of the previous round
    :param g1_returns: Dictionary that maps agents to the G1 returns from an offline rollout (assessed using the G1 value function for this round)
    :param g2_returns: Dictionary that maps agents to the G2 returns from an offline rollout (assessed using the G1 value function for this round)
    :returns Updated values for lambda1 and lambda2
    """

    # Calculate the accumulated returns (averaged over all agents)
    num_agents = len(g1_returns.keys()) # TODO: add check to make sure g1 and g2 num agents the same
    avg_cumulative_g1_returns = torch.sum(torch.tensor(list(g1_returns.values()))).detach().numpy() / num_agents
    avg_cumulative_g2_returns = torch.sum(torch.tensor(list(g2_returns.values()))).detach().numpy() / num_agents

    # Lambda learning rate # TODO: config
    n = 0.1
    # n = 0.01
    # n = 0.00001
    # n = 1

    g1_improvment_rate = (avg_cumulative_g1_returns-avg_cumulative_g1_returns_prev)/avg_cumulative_g1_returns_prev
    g2_improvment_rate = (avg_cumulative_g2_returns-avg_cumulative_g2_returns_prev)/avg_cumulative_g2_returns_prev

    exp_g1_returns = np.exp(n * g1_improvment_rate)
    exp_g2_returns = np.exp(n * g2_improvment_rate)

    lambda_1 = lambda_1_prev * exp_g1_returns / (lambda_1_prev * exp_g1_returns + lambda_2_prev * exp_g2_returns)
    lambda_2 = lambda_2_prev * exp_g2_returns / (lambda_1_prev * exp_g1_returns + lambda_2_prev * exp_g2_returns)

    # Don't let lambda be 0.0
    lambda_1 = max(0.00001, lambda_1)
    lambda_2 = max(0.00001, lambda_2)

    return lambda_1, lambda_2, avg_cumulative_g1_returns, avg_cumulative_g2_returns
# ---------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Parse the configuration file for experiment configuration parameters
    parser = MARLConfigParser()
    args = parser.parse_args()

    if not args.seed:
        args.seed = int(datetime.now())

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    #device = 'cpu'
    print(f" > DEVICE: {device}")
    analysis_steps = args.analysis_steps                        # Defines which checkpoint will be loaded into the Q model
    parameter_sharing_model = args.parameter_sharing_model      # Flag indicating if we're loading a model from DQN with PS
    nn_queue_dir = f"{args.nn_queue_directory}"                 # Name of directory containing the stored queue model nn from training
    nn_speed_overage_dir = f"{args.nn_speed_overage_directory}" # Name of directory containing the stored speed overage model nn from training

    print(f"  > nn_queue_directory: {nn_queue_dir}")
    print(f"  > nn_speed_overage_dir: {nn_speed_overage_dir}")

    # Initialize directories for logging, note that that models will be saved to subfolders created in the directory that was used to 
    # generate the dataset
    experiment_time = str(datetime.now()).split('.')[0].replace(':','-')
    experiment_name = "{}__N{}__exp{}__seed{}__{}".format(args.gym_id, args.N, args.exp_name, args.seed, experiment_time)
    save_dir = f"batch_offline_RL_logs/{experiment_name}"
    csv_save_dir = f"{save_dir}/csv"
    os.makedirs(f"{save_dir}/policies/mean")
    os.makedirs(f"{save_dir}/constraints/queue/mean")
    os.makedirs(f"{save_dir}/constraints/speed_overage/mean")
    os.makedirs(csv_save_dir)

    print("  > Parameter Sharing Enabled: {}".format(parameter_sharing_model))

    # Create the env
    # Sumo must be created using the sumo-rl module
    # NOTE: we have to use the parallel env here to conform to this implementation of dqn
    if (args.sumo_reward != "queue"):
        print(f"  > WARNING: Reward '{args.sumo_reward}' specified but being ignored")
    print(f"  > Setting up environment with standard 'queue' reward")
    print(f"    > This is to ensure that the 'g1' constraint always corresponds to speed threshold and the 'g2' constraint corresponds to queue length")
    # The 'queue' reward is being used here which returns the (negative) total number of vehicles stopped at all intersections
    env = sumo_rl.parallel_env(net_file=args.net, 
                            route_file=args.route,
                            use_gui=args.sumo_gui,
                            max_green=args.max_green,
                            min_green=args.min_green,
                            num_seconds=args.sumo_seconds,  # TODO: for some reason, the env is finishing after 1000 seconds
                            add_system_info=True,   # Default is True
                            add_per_agent_info=True,    # Default is True
                            reward_fn='queue',
                            observation_class=CustomObservationFunction,
                            sumo_warnings=False)

    # Pull some information from env
    agents = env.possible_agents
    num_agents = len(env.possible_agents)
    action_spaces = env.action_spaces
    observation_spaces = env.observation_spaces

    # Initialize dicitonaries
    episode_rewards = {agent: 0 for agent in agents}    # Dictionary that maps the each agent to its cumulative reward each episode

    print("\n=================== Environment Information ===================")
    print(" > agents: {}".format(agents))
    print(" > num_agents: {}".format(num_agents))
    print(" > action_spaces: {}".format(action_spaces))
    print(" > observation_spaces: {}".format(observation_spaces))

    # Construct the Q-Network model. This is the agent that will be used to generate the dataset
    # Note the dimensions of the model varies depending on if the parameter sharing algorithm was used or the normal independent
    # DQN model was used
    list_of_policies = []
    eg_agent = agents[0]
    queue_model_policies = {}  # Dictionary for storing q-networks (maps agent to a q-network)
    speed_overage_model_policies = {}    

    if parameter_sharing_model:
        # In this case, each agent will still have it's own network, they will just be copies of each other

        # Define the shape of the observation space depending on if we're using a global observation or not
        # Regardless, we need to add an array of length num_agents to the observation to account for one hot encoding
        if args.global_obs:
            observation_space_shape = tuple((shape+1) * (num_agents) for shape in observation_spaces[eg_agent].shape)
        else:
            observation_space_shape = np.array(observation_spaces[eg_agent].shape).prod() + num_agents  # Convert (X,) shape from tuple to int so it can be modified
            observation_space_shape = tuple(np.array([observation_space_shape]))                        # Convert int to array and then to a tuple

        # Queue model policies
        queue_model_policy = Actor(observation_space_shape, action_spaces[eg_agent].n).to(device) # In parameter sharing, all agents utilize the same q-network
        
        # Load the Q-network file
        nn_queue_file = "{}/{}.pt".format(nn_queue_dir, analysis_steps)
        queue_model_policy.load_state_dict(torch.load(nn_queue_file))
        print(" > Loading NN from file: {} for 'queue' policy".format(nn_queue_file))

        # Speed overage model policies
        speed_overage_model_policy = Actor(observation_space_shape, action_spaces[eg_agent].n).to(device) # In parameter sharing, all agents utilize the same q-network

        # Load the Q-network file
        nn_speed_overage_file = "{}/{}.pt".format(nn_speed_overage_dir, analysis_steps)
        speed_overage_model_policy.load_state_dict(torch.load(nn_speed_overage_file))
        print(" > Loading NN from file: {} for 'speed threshold' policy".format(nn_speed_overage_file))

        for agent in agents: 
            speed_overage_model_policies[agent] = speed_overage_model_policy
            queue_model_policies[agent] = queue_model_policy


    # Else the agents were trained using normal independent DQN so each agent gets its own Q-network model
    else: 
        
        # Load the Q-Network NN model for each agent from the specified anaylisis checkpoint step from training
        for agent in agents:
            observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if args.global_obs else observation_spaces[agent].shape
            queue_model_policies[agent] = Actor(observation_space_shape, action_spaces[agent].n)
            speed_overage_model_policies[agent] = Actor(observation_space_shape, action_spaces[agent].n)

            # Queue model policies
            nn_queue_file = "{}/{}-{}.pt".format(nn_queue_dir, analysis_steps, agent)
            queue_model_policies[agent].load_state_dict(torch.load(nn_queue_file))
            print(" > Loading NN from file: {} for 'queue' policy".format(nn_queue_file))

            # Repeat for speed overage model policies
            nn_speed_overage_file = "{}/{}-{}.pt".format(nn_speed_overage_dir, analysis_steps, agent)
            speed_overage_model_policies[agent].load_state_dict(torch.load(nn_speed_overage_file))
            print(" > Loading NN from file: {} for 'speed threshold' policy".format(nn_speed_overage_file))

    # List of policies is [speed_overage, queue] - the order is imporant
    list_of_policies.append(speed_overage_model_policies)
    list_of_policies.append(queue_model_policies)

    # Seed the env
    env.reset(seed=args.seed)
    for agent in agents:
        action_spaces[agent].seed(args.seed)
        observation_spaces[agent].seed(args.seed)

    # Initialize the env
    obses, _ = env.reset()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    """
    Step 1:
        Generate the dataset using a previously trained model, the dataset should
        contain information about the contstraint functions as well as the observations

        The dataset should also be generated using "optimal" actions (optimal according to the provided
        policy) ~80% of the time (though this number can be changed) and the rest of the time random
        actions should be used
    """

    if (args.dataset_path == ""):
        # No dataset provided

        # Path to save the dataset
        dataset_save_dir = f"{save_dir}/dataset"
        os.makedirs(dataset_save_dir)

        # We need to generate the dataset
        # TODO: add these arguments to config file
        dataset = GenerateDataset(env, 
                              list_of_policies, 
                              exceess_speed_action_ratio=0.4,
                              queue_action_ratio=0.4, 
                              num_episodes=50,   
                              episode_steps=args.sumo_seconds,
                              parameter_sharing_model=args.parameter_sharing_model)
        
        with open(f"{dataset_save_dir}/dataset.pkl", "wb") as f:
            pickle.dump(dataset, f)

    else:
        print(f" > Loading dataset from: {args.dataset_path}")
        # Load the previously generated dataset
        with open(f"{args.dataset_path}", "rb") as f:
            dataset = pickle.load(f)

        print(f" > Dataset size: {len(dataset[eg_agent].buffer)} per agent")

    # Normalize the constraint values in the dataset
    constraint_ratio = 0.25  # TODO: config
    # constraint_ratio = 0.75
    # constraint_ratio = 0.5
    # constraint_ratio = -0.25
    # constraint_ratio = 0.0
        
    # Calculate the upper (i.e. less negative) bound of the g1 constraint 
    # This is the average reward per step (for all agents) of the excessive speed policy when evaluating it according 
    # to the excessive speed (i.e. speed_overage) reward
    print(f" > Calculating g1 constraint upper bound using 'speed overage' policies")
    g1_upper_bound = CalculateAverageRewardPerStep(queue_length_env=env,
                                                   policies_to_use=speed_overage_model_policies,
                                                   reward_to_evaluate='speed_overage',
                                                   config_args=args)
    
    # Calculate the lower (i.e. more negative) bound of the g1 constraint  
    # This is the average reward per step (for all agents) of the queue length policy when evaluating it according 
    # to the excessive speed (i.e. speed_overage) reward  
    print(f" > Calculating g1 constraint lower bound using 'queue' policies")
    g1_lower_bound = CalculateAverageRewardPerStep(queue_length_env=env,
                                                   policies_to_use=queue_model_policies,
                                                   reward_to_evaluate='speed_overage',
                                                   config_args=args)
    print(f" > Applying constraints to dataset")
    print(f"   > g1_upper_bound: {g1_upper_bound}")
    print(f"   > g1_lower_bound: {g1_lower_bound}")
    print(f"   > constraint ratio: {constraint_ratio}")
    normalized_dataset = NormalizeDataset(dataset, 
                                          constraint_ratio=constraint_ratio,
                                          g1_lower_bound=g1_lower_bound,
                                          g1_upper_bound=g1_upper_bound)

    """
    Step 2:
        Use the generated dataset to iteratively learn a new policy
        Essentially we want to evaluate the new policy, E[pi] and the constraint function E[G]
    """
    mean_policies, policies, mean_lambdas, lambdas = OfflineBatchRL(env,
                                                                normalized_dataset,
                                                                list_of_policies,
                                                                args,
                                                                save_dir,
                                                                csv_save_dir,
                                                                max_num_rounds=10)   # Lucky 7


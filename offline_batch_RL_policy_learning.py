"""
offline_batch_RL_policy_learning.py

Description:
    Offline batch RL for learning a policy subject to constraints. The idea here is that we can utilize experiences from various 
    policies to learn a new policy that is optimal according to some objective function and also obeys some secondary constraints. 
    This algorithm is essentially decentralized

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


# Set up the system and environement
# Make sure SUMO env variable is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


def CalculateMaxSpeedOverage(max_speed:float, speed_limit:float) -> float:
    """
    Calculate how much the agents' max speeds exceeded some speed limit
    :param max_speed: Max speed of all cars observed by the agent (assumed at a single step)
    :param speed_limit: User defined threshold over which the overage is calculated
    :returns -1 times how much the max speed exceeded the speed limit
    """

    overage = 0.0

    if (max_speed > speed_limit):
        overage = -1.0*(max_speed - speed_limit)

    return overage


def GenerateDataset(env: sumo_rl.parallel_env, 
                    list_of_policies: list,
                    optimal_action_ratio:float = 0.8, 
                    num_episodes:int=100, 
                    episode_steps:int=1000) -> dict:
    """
    :param env: The sumo environment
    :param list_of_policies: A list of trained neural networks used to generate the dataset by acting in the environment
    :param optimal_action_ratio: Number specifying the fraction of time in which an optimal action should be taken
            e.g. 0.8 of all actions should be "optimal" (according to the provdied q_network), 0.2 of all actions
            will therefore be random
    :param num_episodes: number of episodes to run to populate the dataset
    :param episode_steps: number of steps to take in each episode
    :returns a dictionary that maps each agent to 
    """
    print(f" >> Generating dataset with {num_episodes} episodes of {episode_steps} steps")
    start_time = datetime.now()

    DATASET_SIZE = num_episodes*episode_steps
    SPEED_OVERAGE_THRESHOLD = 13.89
    agents = env.possible_agents

    # Initialize the dataset as a dictionary that maps agents to Dataset objects that are full of experience
    dataset = {agent : Dataset(DATASET_SIZE) for agent in agents}

    # Define empty dictionary tha maps agents to actions
    actions = {agent: None for agent in agents}
    action_spaces = env.action_spaces

    # Define dictionaries to hold the values of the constraints (g1 and g2) each step
    constraint_1 = {agent : 0 for agent in agents}  # Maps each agent to its (-1) * MAX SPEED OVERAGE for this step
    constraint_2 = {agent : 0 for agent in agents}  # Maps each agent to the (-1) * NUBMER OF CARS STOPPED for this step

    for episode in range(num_episodes):
        print(f" >>> Generating Episode: {episode}")

        # Reset the environment
        obses, _ = env.reset()

        for step in range(episode_steps):

            # Set the action for each agent
            for agent in agents:
                random_number = random.random()
                if (random_number < (1-optimal_action_ratio)):
                    actions[agent] = action_spaces[agent].sample()
                else:

                    # Give equal probability to each of the provided policies
                    q_network = random.choice(list_of_policies)

                    # Actor choses the actions
                    action, _, _ = q_network[agent].to(device).get_action(obses[agent])
                    actions[agent] = action.detach().cpu().numpy()

            # Apply all actions to the env
            next_obses, rewards, dones, truncated, info = env.step(actions)

            if np.prod(list(dones.values())):
                # Start the next episode
                print(f" >>> Episode complete at {step} steps, going to next episode")
                break

            # Caclulate constraints and add the experience to the dataset
            for agent in agents:
                 max_speed_observed_by_agent = next_obses[agent][-1]
                 constraint_1[agent] = CalculateMaxSpeedOverage(max_speed_observed_by_agent, SPEED_OVERAGE_THRESHOLD)
                 constraint_2[agent] = rewards[agent]   # NOTE: This assumes that the environment was configured with the "queue" reward
                 dataset[agent].put((obses[agent], actions[agent], next_obses[agent], constraint_1[agent], constraint_2[agent], dones[agent]))

            obses = next_obses

    stop_time = datetime.now()
    print(" >> Dataset generation complete")
    print(" >> Total execution time: {}".format(stop_time-start_time))

    env.close()

    return dataset

def NormalizeDataset(dataset:dict,
                     constraint_ratio:float=0.25) -> dict:
    """
    Function for normalizing the dataset so that the values of one constraint do not normalize the other
    :param dataset: Dictionary that maps agents to a dataset of experience tuples
    :param constraint_ratio: The ratio to use in the weight adjustment, applied positive to g1 constraint and negative to g2 constraint
    :returns Dictionary that maps agents to normalized datasets
    """
    print(f" > Normalizing dataset constraint values")
    # Intialize dataset
    normalized_dataset = {}

    total_c_g1 = 0.0
    total_c_g2 = 0.0

    for agent in dataset.keys():
        
        # Initialize constraint bounds
        C_g1 = 0.0
        C_g2 = 0.0
        
        G_1 = 0.0
        G_2 = 0.0

        normalized_g1s = []
        normalized_g2s = []

        # Number of experience tuples for this agent
        n = len(dataset[agent].buffer)
        print(f" > Agent '{agent}' buffer size: {n} ")

        normalized_dataset[agent] = Dataset(n)
        
        # Calculate the bounds for the g1 and g2 constraints
        for i in range(n):
            s_obses, s_actions, s_next_obses, s_g1s, s_g2s, s_dones = dataset[agent].buffer[i]
            C_g1 += s_g1s
            C_g2 += s_g2s

        C_g1 = C_g1 / n * (1.0 + constraint_ratio)
        # C_g2 = C_g2 / n * (1.0 - constraint_ratio)    

        # Now apply the bounds to the g1 and g2 observations
        for i in range(n):
            s_obses, s_actions, s_next_obses, s_g1s, s_g2s, s_dones = dataset[agent].buffer[i]
            
            g1_normalized = min(C_g1, s_g1s) - C_g1
            normalized_g1s.append(g1_normalized)
            G_1 += g1_normalized

            # g2_normalized = min(C_g2, s_g2s) - C_g2   
            g2_normalized = s_g2s # No constraint for g2
            normalized_g2s.append(g2_normalized)
            G_2 += g2_normalized
            
        for i in range(n):
            s_obses, s_actions, s_next_obses, _, _, s_dones = dataset[agent].buffer[i]
            
            # Calculate normalized g1 and g2
            # NOTE: weirdly, the best response we have seen (both g1 and g2 reaching maximal returns) occured 
            # without this step... 
            # Not sure why
            g1_n = normalized_g1s[i]/abs(G_1)
            if g1_n > 0.0:
                print(f"ERROR: normalized g1 = {g1_n}, algorithm assumes g1 < 0")
                sys.exit()

            g2_n = normalized_g2s[i]/abs(G_2)
            if g2_n > 0.0:
                print(f"ERROR: normalized g2 = {g2_n}, algorithm assumes g2 < 0")
                sys.exit()
            
        

            # Add it to the new dataset
            normalized_dataset[agent].put((s_obses, s_actions, s_next_obses, g1_n, g2_n, s_dones))

        total_c_g1 += C_g1
        total_c_g2 += C_g2
    print(f" >>>> total_c_g1 = {total_c_g1}    total_c_g2 = {total_c_g2}")

    return normalized_dataset


def OfflineBatchRL(env:sumo_rl.parallel_env,
                    dataset: dict,
                    dataset_policies:list,
                    config_args,
                    nn_save_dir:str,
                    csv_save_dir:str,
                    max_num_rounds:int=10) -> (dict, dict, list, list):
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

    # Initialize the mean networks and the rollout datasets
    prev_mean_policies = {}
    prev_g1_constraints = {}
    prev_g2_constraints = {}
    rollout_mini_dataset = {}
    num_agents = len(agents) 
    for agent in agents:
        observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if config_args.global_obs else observation_spaces[agent].shape
        prev_mean_policies[agent] = Actor(observation_space_shape, action_spaces[agent].n).to(device)
        prev_g1_constraints[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
        prev_g2_constraints[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
        # rollout_mini_dataset[agent] = dataset[agent].sample(1000) # TODO: config  
        rollout_mini_dataset[agent] = dataset[agent].sample(50000)

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
        G1_pi = FittedQEvaluation(observation_spaces, 
                                    action_spaces, 
                                    agents,
                                    policies,
                                    dataset, 
                                    csv_save_dir,
                                    mean_constraint_suffixes[0],
                                    config_args, 
                                    constraint="speed_overage")
        
        # Save the value function every round
        # Format is constraint_<round>-<agent>.pt
        for a in agents:
            torch.save(G1_pi[a].state_dict(), f"{nn_save_dir}/constraints/speed_overage/constraint_{t}-{a}.pt")        
        

        # Evaluate G_2^pi 
        G2_pi = FittedQEvaluation(observation_spaces, 
                                    action_spaces, 
                                    agents,
                                    policies,
                                    dataset, 
                                    csv_save_dir,
                                    mean_constraint_suffixes[1],
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
                                                      mean_constraint_suffixes[0],
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
                                                      mean_constraint_suffixes[1],
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
        print(f">>> EVALUATING G1_pi IN OFFLINE ROLLOUT")
        g1_returns = PerformOfflineRollout(G1_pi, policies, rollout_mini_dataset)

        print(f">>> EVALUATING G2_pi IN OFFLINE ROLLOUT")
        g2_returns = PerformOfflineRollout(G2_pi, policies, rollout_mini_dataset)
        # TODO: perform some kind of similar rollout for the dataset policies to compare?


        # NOTE: We are logging the lambda values produced by round X rather than the values used in 
        # round X 
        # Adjust lambda
        lambda_1, lambda_2 = OnlineLambdaLearning(lambda_1, lambda_2, g1_returns, g2_returns)

        if (t > 1):
            # On the first round, the expected value of lambda was set as the initial value of lambda
            mean_lambda_1 = 1/t*(lambda_1 + ((t-1) * mean_lambda_1))
            mean_lambda_2 = 1/t*(lambda_2 + ((t-1) * mean_lambda_2))

        # Save the rollout returns and the updated lambdas
        with open(f"{csv_save_dir}/rollouts.csv", "a", newline="") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=['round'] + [agent + '_return_g1' for agent in agents] + 
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

        # Now run some online rollouts to compare performance between the learned policy and the dataset policy
        print(f" >> Performing online rollout for mean policy")
        mean_policy_sys_return, mean_policy_g1_return, mean_policy_g2_return = PerformRollout(env, prev_mean_policies, config_args)
        
        print(f" >> Performing online rollout for current learned policy")
        current_policy_sys_return, current_policy_g1_return, current_policy_g2_return = PerformRollout(env, policies, config_args)
        
        # TODO: make this more generic
        threshold_policy = dataset_policies[0]
        print(f" >> Performing online rollout for speed overage policy")
        threshold_policy_sys_return, threshold_policy_g1_return, threshold_policy_g2_return = PerformRollout(env, threshold_policy, config_args)
        
        queue_policy = dataset_policies[1]
        print(f" >> Performing online rollout for queue policy")
        queue_policy_sys_return, queue_policy_g1_return, queue_policy_g2_return = PerformRollout(env, queue_policy, config_args)

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

    :param observation_spaces: Dictionary that maps an agent to the observation space dimensions
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


    print(" >> Beginning Fitted Q Iteration")
    start_time = datetime.now()

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
    eval_obses, eval_actions, eval_next_obses, eval_g1s, eval_g2s, eval_dones = dataset[agents[0]].sample(1)
    
    for global_step in range(config_args.total_timesteps):

        if config_args.global_obs:
            print("ERROR: global observations not supported for FittedQIteration")
            return {}

        # Training for each agent
        for agent in agents:

            # TODO: remove?
            if (global_step % config_args.train_frequency == 0):  
                
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
                if global_step % args.target_network_frequency == 0:
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


    print(" >> Beginning Fitted Q Evaluation")
    start_time = datetime.now()

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
    eval_obses, eval_actions, eval_next_obses, eval_g1s, eval_g2s, eval_dones = dataset[agents[0]].sample(1)

    # TODO: this should be updated to be for k = 1:K (does not need to be the same as total_timesteps)
    for global_step in range(config_args.total_timesteps):

        if config_args.global_obs:
            print("ERROR: global observations not supported for FittedQEvaluation")
            return {}

        # Training for each agent
        for agent in agents:
            
            if (global_step % config_args.train_frequency == 0):  
                
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

    mean_policy = {}    # Dictionary that maps agents to the "mean" policy for this round
    optimizer = {}      # Dictionary for storing optimizer for each agent's network
    losses = {}         # Dictionary that maps each agent to the loss values for its network
    num_agents = len(agents)
    for agent in agents: 
        observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if config_args.global_obs else observation_spaces[agent].shape

        mean_policy[agent] = Actor(observation_space_shape, action_spaces[agent].n).to(device)

        # Initialize the mean policy using the previous one
        mean_policy[agent].load_state_dict(previous_mean_policy[agent].state_dict())

        optimizer[agent] = optim.Adam(mean_policy[agent].parameters(), lr=config_args.learning_rate) # All agents use the same optimizer for training
        losses[agent] = None

    loss_fn = nn.MSELoss() # TODO: should this be MSE or Cross Entropy?

    # For k = 1:K
    for global_step in range(config_args.total_timesteps):

        # Training for each agent
        for agent in agents:

            if (global_step % config_args.train_frequency == 0):  
                
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

    mean_constraint = {}    # Dictionary that maps agents to the "mean" constraint value function for this round
    optimizer = {}          # Dictionary for storing optimizer for each agent's network
    losses = {}             # Dictionary that maps each agent to the loss values for its network

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

        # Training for each agent
        for agent in agents:

            # TODO: remove?
            if (global_step % config_args.train_frequency == 0):  
                
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
    
    return mean_constraint


def PerformRollout(env:sumo_rl.parallel_env, policy:dict, config_args)->(dict, dict, dict):
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

    # This function assumes that the environment is set up with the 'queue' reward so if that's not the case
    # we need to return an error
    if (config_args.sumo_reward != 'queue'):
        print(f"ERROR: Cannot evaluate constraints while performing rollout with environment configured with '{config_args.sumo_reward}'")
        return {}, {}, {}

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

    # Perform the rollout
    for sumo_step in range(config_args.sumo_seconds):
        # Populate the action dictionary
        for agent in agents:
            # Only use optimal actions according to the policy
            action, _, _ = policy[agent].to(device).get_action(obses[agent])
            actions[agent] = action.detach().cpu().numpy()

        # Apply all actions to the env
        next_obses, rewards, dones, truncated, info = env.step(actions)


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
        
        if np.prod(list(dones.values())):
            return episode_rewards, episode_constraint_1, episode_constraint_2
        
    return episode_rewards, episode_constraint_1, episode_constraint_2


def PerformOfflineRollout(value_function:dict, policies:dict, mini_dataset:dict)->dict:
    """
    :param value_function: Dictionary that maps agents to the learned value function for a given constraint
    :param mini_dataset: A subset of the larger dataset that contains experience tuples that are evaluated in the rollout
            using the provided dataset
            NOTE: This function assumes the dataset has been sampled such that mini_dataset contains lists of of experience tuples for each agent
    :returns The cummulative return for all agents of the mini dataset according to the provided value functions normalized by the size of the mini dataset
    """
    
    agents = list(value_function.keys())
    cumulative_return = {agent:0.0 for agent in agents}

    for agent in agents:
        obses_array, _, _, _, _, _ = mini_dataset[agent]

        # Get the max_a Q(s,a) for the observation
        actions_from_policy, _, _ = policies[agent].to(device).get_action(obses_array)
        
        # Evaluate the actions that were selected by the policies from this observation
        values = (value_function[agent].forward(obses_array)).to(device).gather(1, actions_from_policy.view(-1,1)).squeeze()
        
        # Add up the max values of all states from the mini dataset
        cumulative_return[agent] = sum(values)/(len(mini_dataset[agent]))
        print(f" >>>> cumulative_return for agent '{agent}': {cumulative_return[agent]}")

    return cumulative_return


def OnlineLambdaLearning(lambda_1_prev:float, 
                         lambda_2_prev:float, 
                         g1_returns:dict, 
                         g2_returns:dict)->(float, float):
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
    n = 0.001 

    exp_g1_returns = np.exp(n * avg_cumulative_g1_returns)
    exp_g2_returns = np.exp(n * avg_cumulative_g2_returns)

    lambda_1 = lambda_1_prev * exp_g1_returns / (lambda_1_prev * exp_g1_returns + lambda_2_prev * exp_g2_returns)
    lambda_2 = lambda_2_prev * exp_g2_returns / (lambda_1_prev * exp_g1_returns + lambda_2_prev * exp_g2_returns)

    # Don't let lambda be 0.0
    lambda_1 = max(0.00001, lambda_1)
    lambda_2 = max(0.00001, lambda_2)

    return lambda_1, lambda_2

# ---------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Parse the configuration file for experiment configuration parameters
    parser = MARLConfigParser()
    args = parser.parse_args()

    if not args.seed:
        args.seed = int(datetime.now()) 

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    #device = 'cpu'
    print(f"DEVICE: {device}")
    analysis_steps = args.analysis_steps                        # Defines which checkpoint will be loaded into the Q model
    parameter_sharing_model = args.parameter_sharing_model      # Flag indicating if we're loading a model from DQN with PS
    nn_queue_dir = f"{args.nn_queue_directory}"                 # Name of directory containing the stored queue model nn from training
    nn_speed_overage_dir = f"{args.nn_speed_overage_directory}" # Name of directory containing the stored speed overage model nn from training
    
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

    print(" > Parameter Sharing Enabled: {}".format(parameter_sharing_model))

    # Create the env
    # Sumo must be created using the sumo-rl module
    # Note we have to use the parallel env here to conform to this implementation of dqn
    # The 'queue' reward is being used here which returns the (negative) total number of vehicles stopped at all intersections
    if (args.sumo_reward == "custom"):
        # Use the custom "max speed" reward function
        print ( " > Setting up environment with CUSTOM reward")
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
        print ( " > Setting up environment with standard reward: {}".format(args.sumo_reward))
        # The 'queue' reward is being used here which returns the (negative) total number of vehicles stopped at all intersections
        env = sumo_rl.parallel_env(net_file=args.net, 
                                route_file=args.route,
                                use_gui=args.sumo_gui,
                                max_green=args.max_green,
                                min_green=args.min_green,
                                num_seconds=args.sumo_seconds,  # TODO: for some reason, the env is finishing after 1000 seconds
                                add_system_info=True,
                                add_per_agent_info=True,
                                reward_fn=args.sumo_reward,
                                observation_class=CustomObservationFunction,
                                sumo_warnings=False)

    # Pull some information from env
    agents = env.possible_agents
    num_agents = len(env.possible_agents)
    action_spaces = env.action_spaces
    observation_spaces = env.observation_spaces

    # Initialize dicitonaries
    onehot_keys = {agent: i for i, agent in enumerate(agents)}
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
    if parameter_sharing_model:
        # Define the shape of the observation space depending on if we're using a global observation or not
        # Regardless, we need to add an array of length num_agents to the observation to account for one hot encoding
        if args.global_obs:
            observation_space_shape = tuple((shape+1) * (num_agents) for shape in observation_spaces[eg_agent].shape)
        else:
            observation_space_shape = np.array(observation_spaces[eg_agent].shape).prod() + num_agents  # Convert (X,) shape from tuple to int so it can be modified
            observation_space_shape = tuple(np.array([observation_space_shape]))                        # Convert int to array and then to a tuple

        # Queue model policies
        queue_model_policies = Actor(observation_space_shape, action_spaces[eg_agent].n, parameter_sharing_model).to(device) # In parameter sharing, all agents utilize the same q-network
        
        # Load the Q-network file
        nn_queue_file = "{}/{}.pt".format(nn_queue_dir, analysis_steps)
        queue_model_policies.load_state_dict(torch.load(nn_queue_file))

        # Speed overage model policies
        speed_overage_model_policies = Actor(observation_space_shape, action_spaces[eg_agent].n, parameter_sharing_model).to(device) # In parameter sharing, all agents utilize the same q-network

        # Load the Q-network file
        nn_speed_overage_file = "{}/{}.pt".format(nn_speed_overage_dir, analysis_steps)
        speed_overage_model_policies.load_state_dict(torch.load(nn_speed_overage_file))


    # Else the agents were trained using normal independent DQN so each agent gets its own Q-network model
    else: 
        
        queue_model_policies = {}  # Dictionary for storing q-networks (maps agent to a q-network)
        speed_overage_model_policies = {}

        # Load the Q-Network NN model for each agent from the specified anaylisis checkpoint step from training
        for agent in agents: 
            observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if args.global_obs else observation_spaces[agent].shape
            queue_model_policies[agent] = Actor(observation_space_shape, action_spaces[agent].n)
            speed_overage_model_policies[agent] = Actor(observation_space_shape, action_spaces[agent].n)

            # Queue model policies
            nn_queue_file = "{}/{}-{}.pt".format(nn_queue_dir, analysis_steps, agent) 
            queue_model_policies[agent].load_state_dict(torch.load(nn_queue_file))
            print(" > Loading NN from file: {} for dataset generation".format(nn_queue_file))

            # Repeat for speed overage model policies
            nn_speed_overage_file = "{}/{}-{}.pt".format(nn_speed_overage_dir, analysis_steps, agent) 
            speed_overage_model_policies[agent].load_state_dict(torch.load(nn_speed_overage_file))
            print(" > Loading NN from file: {} for dataset generation".format(nn_speed_overage_file))


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
        policy) ~80% of the time and the rest of the time random actions should be used
    """

    if (args.dataset_path == ""):

        dataset_save_dir = f"{save_dir}/dataset"
        os.makedirs(dataset_save_dir)
    
        # We need to generate the dataset
        dataset = GenerateDataset(env, 
                              list_of_policies, 
                              optimal_action_ratio=0.8, 
                              num_episodes=50,   
                              episode_steps=args.sumo_seconds)
        

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
    # constraint_ratio = -0.25
    # constraint_ratio = 0.0
    normalized_dataset = NormalizeDataset(dataset, constraint_ratio=constraint_ratio)

    """
    Step 2:
        Use the generated dataset to iteratively learn a new policy 
        Essentially we want to evaluate the new policy, E[pi] and the constraint function E[G]
    """
    perform_rollout_comparisons = True
    mean_policies, policies, mean_lambdas, lambdas = OfflineBatchRL(env,
                                                                normalized_dataset, 
                                                                list_of_policies,
                                                                args, 
                                                                save_dir,
                                                                csv_save_dir,
                                                                max_num_rounds=10)   # Lucky 7 


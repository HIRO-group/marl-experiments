"""
offline-batch-RL-policy-learning.py

Description:


Usage:


References:


"""

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from datetime import datetime
import random
import csv

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
# from linear_schedule import LinearSchedule
# from ensemble_weighted_network import EnsembleWeightedNetwork


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
                    q_network: dict, 
                    optimal_action_ratio:float = 0.8, 
                    num_episodes:int=100, 
                    episode_steps=1000) -> dict:
    """
    :param env: The sumo environment
    :param q_netowrk: The trained neural network used to generate the dataset by acting in the environment
    :param optimal_action_ratio: Number specifying the fraction of time in which an optimal action should be taken
            e.g. 0.8 of all actions should be "optimal" (according to the provdied q_network), 0.2 of all actions
            will therefore be random
    :param num_episodes: number of episodes to run to populate the dataset
    :param episode_steps: number of steps to take in each episode
    :returns a dictionary that maps each agent to 
    """
    print(" >> Generating dataset")
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
                if (random.random() < (1-optimal_action_ratio)):
                    actions[agent] = action_spaces[agent].sample()
                else:
                    # Actor choses the actions
                    action, _, _ = q_network[agent].get_action(obses[agent])
                    actions[agent] = action.detach().cpu().numpy()

            # Apply all actions to the env
            next_obses, rewards, dones, truncated, info = env.step(actions)

            if np.prod(list(dones.values())):
                # Start the next episode
                break

            # Caclulate constraints and add the experience to the dataset
            for agent in agents:
                 max_speed_observed_by_agent = next_obses[agent][-1]
                 constraint_1[agent] = CalculateMaxSpeedOverage(max_speed_observed_by_agent, SPEED_OVERAGE_THRESHOLD)
                 constraint_2[agent] = rewards[agent]
                 dataset[agent].put((obses[agent], actions[agent], next_obses[agent], constraint_1[agent], constraint_2[agent], dones[agent]))

            obses = next_obses

    stop_time = datetime.now()
    print(" >> Dataset generation complete")
    print(" >> Total execution time: {}".format(stop_time-start_time))

    env.close()

    return dataset


def OfflineBatchRL(env:sumo_rl.parallel_env,
                    dataset: dict,
                    dataset_policy: dict,
                    perform_rollout_comparisons:bool,
                    config_args,
                    nn_save_dir:str,
                    csv_save_dir:str,
                    constraint:str="") -> (dict, dict):

    """
    Perform offline batch reinforcement learning
    Here we use a provided dataset to learn and evaluate a policy for a given number of "rounds"
    Each round, a policy is learned and then evaluated (each of which involves solving an RL problem). 
    The provided constraint function defines how the "target" is calculated for each RL problem. At the end of the 
    round, the expected value of the polciy and the value function is calculated.

    :param observation_spaces: Dictionary that maps an agent to the observation space dimensions
    :param action_spaces: Dictionary that maps agent to its action space
    :param agents: List of agent names
    :param dataset: Dictionary that maps each agent to its experience tuple
    :param config_args: Configuration arguments used to set up the experiment
    :param nn_save_dir: Directory in which to save the models each round
    :pram csv_save_dir: Directory in which to save the csv file 
    :param constraint: 'speed_overage' or 'queue', defines how the target should be determined while learning the policy
    :returns A dictionary that maps each agent to its learned policy
    """

    print(f" > Performing batch offline reinforcement learning")
    function_start_time = datetime.now()

    # Check inputs
    if not ((constraint == "queue") or (constraint == "speed_overage")):
        print(f"ERROR: Constraint function '{constraint}' not recognized, unable to perform Offline Batch RL")
        return {}, {}
    else:
        print(f">> Constraint '{constraint}' recognized!")

    # TODO: could make these configs
    MAX_NUM_ROUNDS = 10
    # OMEGA = 0.1   # TODO: we don't know what this should be yet

    agents = env.possible_agents
    action_spaces = env.action_spaces
    observation_spaces = env.observation_spaces

    # Initialize csv files
    with open(f"{csv_save_dir}/batch_offline_RL_expectation_pi.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['omega', 'round'])
        csv_writer.writeheader()

    with open(f"{csv_save_dir}/batch_offline_RL_expectation_g.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['omega', 'round'])
        csv_writer.writeheader()

    with open(f"{csv_save_dir}/FQE_loss.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['Q(s,a) Sample','global_step'])
        csv_writer.writeheader()

    with open(f"{csv_save_dir}/FQI_actor_loss.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['Pi(a|s) Sample', 'global_step'])
        csv_writer.writeheader()

    # Create csv files if we plan to compare the learned policy to the dataset policy each round
    # Each round, we will store the return for both constraints for the learned policy and the dataset policy
    if perform_rollout_comparisons:
        with open(f"{csv_save_dir}/rollout_pi_d_constraint_1.csv", "w", newline="") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['system_episode_constraint_1', 'round'])
            csv_writer.writeheader()

        with open(f"{csv_save_dir}/rollout_pi_d_constraint_2.csv", "w", newline="") as csvfile: 
            csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['system_episode_constraint_2', 'round'])
            csv_writer.writeheader()

        with open(f"{csv_save_dir}/rollout_pi_t_constraint_1.csv", "w", newline="") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['system_episode_constraint_1', 'round'])
            csv_writer.writeheader()

        with open(f"{csv_save_dir}/rollout_pi_t_constraint_2.csv", "w", newline="") as csvfile: 
            csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['system_episode_constraint_2', 'round'])
            csv_writer.writeheader()


    # Initialize EnsembleWeightedNetwork objects for calculating the expectations each round
    agents_ensemble_pi = {agent : [] for agent in agents}   # Dictionary mapping agents to list of their learned policies each round
    agents_ensemble_g2 = {agent : [] for agent in agents}   # Dictionary mapping agents to their learned constraint value functions each round

    for t in range(MAX_NUM_ROUNDS):
        print(f" >> BEGINNING ROUND: {t}")
        round_start_time = datetime.now()
        # Learn a policy that optimizes actions for the "g2" constraint
        # This is essentially the "actor", policies here are represented as probability density functions of taking an action given a state
        # policies_name = f"policies_{t}"
        policies = FittedQIteration(observation_spaces, 
                                    action_spaces, 
                                    agents,
                                    dataset, 
                                    csv_save_dir,
                                    config_args, 
                                    constraint=constraint) 
        # TODO: save the policy here and add ability to load?
        # Save the policy every round
        for a in agents:
            torch.save(policies[a].state_dict(), f"{nn_save_dir}/policies/policy_{t}-{a}.pt")

        # Evaluate G_2^pi 
        # This is essentially the "critic"
        G2_pi = FittedQEvaluation(observation_spaces, 
                                    action_spaces, 
                                    agents,
                                    policies,
                                    dataset, 
                                    csv_save_dir,
                                    config_args, 
                                    constraint=constraint)
        # Save the value function every round
        for a in agents:
            torch.save(G2_pi[a].state_dict(), f"{nn_save_dir}/constraints/constraint_{t}-{a}.pt")        
        
        # Update the ensembles for each agent by adding the latest policy and g2 value function from this round
        for agent in agents:
            agent_latest_policy = policies[agent]
            agents_ensemble_pi[agent].append(agent_latest_policy)

            agent_latest_g2 = G2_pi[agent]
            agents_ensemble_g2[agent].append(agent_latest_g2)

        # Calculate 1/t*(pi + t-1(E[pi])) for each agent
        expectation_pi = EvaluatePolicyEnsemble(agents_ensemble_pi, dataset, config_args) # TODO: could we replace this with a rollout?

        # Calculate 1/t*(g2 + t-1(E[g2])) for each agent
        expectation_g2 = EvaluateConstraintEnsemble(agents_ensemble_g2, dataset, config_args) # TODO: could we replace this with a rollout?

        # for agent in agents:
        #     print(f" >>> Agent '{agent}'")
        #     print(f" >>>> E[pi]: {expectation_pi[agent]}")
        #     print(f" >>>> E[g2]: {expectation_g2[agent]}")

        # Evaluate difference but don't use it for an exit condition (yet) because we don't know what 
        # OMEGA should be set to
        # Note that we are calculating omega for each agent first then taking the norm of the "vector" of omegas
        omega_dict = {}
        for agent in agents:
            agent_omega = torch.linalg.vector_norm(expectation_pi[agent] - expectation_g2[agent])
            omega_dict[agent] = agent_omega.item()  # Store the values as floats in the dict

        # Convert the omega_dict values to list then to a tensor and then take the norm of it
        omega = torch.linalg.vector_norm(torch.tensor(list(omega_dict.values()))).item()

        # Log the data from this round
        with open(f"{csv_save_dir}/batch_offline_RL_expectation_pi.csv", "a", newline="") as csvfile:    
            csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['omega', 'round'])                        
            csv_writer.writerow({**expectation_pi, **{'omega': omega, 'round': t}})

        with open(f"{csv_save_dir}/batch_offline_RL_expectation_g.csv", "a", newline="") as csvfile:    
            csv_writer = csv.DictWriter(csvfile, fieldnames=agents+['omega', 'round'])                        
            csv_writer.writerow({**expectation_g2, **{'omega': omega, 'round': t}})

        # At the end of each round, compare the latest policy to the one that was used to generate the dataset
        if perform_rollout_comparisons:
            print(f" >> Performing rollout comparison between learned policy and dataset policy")
            
            # Run the rollout on the dataset policy
            episode_rewards_pi_d, episode_constraint_1_pi_d, episode_constraint_2_pi_d = PerformRollout(env, dataset_policy, config_args)
            
            # Add it all up
            system_episode_reward_pi_d = sum(list(episode_rewards_pi_d.values())) # Accumulated reward of all agents
            system_episode_constraint_1_pi_d = sum(list(episode_constraint_1_pi_d.values())) 
            system_episode_constraint_2_pi_d = sum(list(episode_constraint_2_pi_d.values())) 
            
            # Log the data for the dataset policy
            with open(f"{csv_save_dir}/rollout_pi_d_constraint_1.csv", "a", newline="") as csvfile: 
                csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['system_episode_constraint_1', 'round'])
                csv_writer.writerow({**episode_constraint_1_pi_d, **{'system_episode_constraint_1' : system_episode_constraint_1_pi_d, 'round' : t}})
            
            with open(f"{csv_save_dir}/rollout_pi_d_constraint_2.csv", "a", newline="") as csvfile: 
                csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['system_episode_constraint_2', 'round'])
                csv_writer.writerow({**episode_constraint_2_pi_d, **{'system_episode_constraint_2' : system_episode_constraint_2_pi_d, 'round' : t}})


            # Run the rollout on the policy learned this round
            episode_rewards_pi_t, episode_constraint_1_pi_t, episode_constraint_2_pi_t = PerformRollout(env, policies, config_args)

            # Add it all up
            system_episode_reward_pi_t = sum(list(episode_rewards_pi_t.values())) # Accumulated reward of all agents
            system_episode_constraint_1_pi_t = sum(list(episode_constraint_1_pi_t.values())) 
            system_episode_constraint_2_pi_t = sum(list(episode_constraint_2_pi_t.values())) 

            with open(f"{csv_save_dir}/rollout_pi_t_constraint_1.csv", "a", newline="") as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['system_episode_constraint_1', 'round'])
                csv_writer.writerow({**episode_constraint_1_pi_t, **{'system_episode_constraint_1' : system_episode_constraint_1_pi_t, 'round' : t}})

            with open(f"{csv_save_dir}/rollout_pi_t_constraint_2.csv", "a", newline="") as csvfile: 
                csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['system_episode_constraint_2', 'round'])
                csv_writer.writerow({**episode_constraint_2_pi_t, **{'system_episode_constraint_2' : system_episode_constraint_2_pi_t, 'round' : t}})

            # print(f" >>>>> episode_constraint_1_pi_t: {episode_constraint_1_pi_t}")
            # print(f" >>>>> episode_constraint_2_pi_t: {episode_constraint_2_pi_t}")


        round_completeion_time = datetime.now()
        print(f" >> Round {t} complete!")
        print(f" >> Evaluated Omega: {omega}")
        print(f" >> Round execution time: {round_completeion_time-round_start_time}")

    function_stop_time = datetime.now()
    print(f" > Batch offline reinforcement learning complete")
    print(f" > Total execution time: {function_stop_time-function_start_time}")

    return expectation_pi, expectation_g2


def FittedQIteration(observation_spaces:dict,
                     action_spaces:dict,
                     agents:list,
                     dataset:dict, 
                     csv_save_dir:str,
                     config_args, 
                     constraint:str="") -> dict:
    """
    Implementation of Fitted Q Iteration with function approximation for offline learning of a policy
    Note that this implementation utilizes an "actor-critic" approach to solve the RL problem
    (algorithm 4 from Le, et. al)

    :param observation_spaces: Dictionary that maps an agent to the observation space dimensions
    :param action_spaces: Dictionary that maps agent to its action space
    :param agents: List of agent names
    :param dataset: Dictionary that maps each agent to its experience tuple
    :param csv_save_dir:
    :param config_args: Configuration arguments used to set up the experiment
    :param constraint: 'speed_overage' or 'queue', defines how the target should be determined while learning the policy
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

            if (global_step % config_args.train_frequency == 0):  
                
                # Sample data from the dataset
                s_obses, s_actions, s_next_obses, s_g1s, s_g2s, s_dones = dataset[agent].sample(config_args.batch_size)
                # print(f"s_obses {s_obses}, s_actions {s_actions}, s_next_obses {s_next_obses}, s_g1s {s_g1s}, s_g2s {s_g2s}, s_dones {s_dones}")
                # Compute the target
                with torch.no_grad():
                    # Calculate min_a Q(s',a)
                    target_min = torch.min(target_network[agent].forward(s_next_obses), dim=1)[0]
                    
                    # Calculate the full TD target 
                    # Note that the target in this Fitted Q iteration implementation depends on the type of constraint we are using to 
                    # learn the policy
                    if (constraint == "speed_overage"):
                        # Use the "g1" constraint
                        td_target = torch.Tensor(s_g1s).to(device) + config_args.gamma * target_min * (1 - torch.Tensor(s_dones).to(device))

                    elif (constraint == "queue"):
                        # Use the "g2" constraint
                        td_target = torch.Tensor(s_g2s).to(device) + config_args.gamma * target_min * (1 - torch.Tensor(s_dones).to(device))

                    else: 
                        print(f"ERROR: Constraint function '{constraint}' not recognized, unable to train using Fitted Q Iteration")

                q_values = q_network[agent].forward(s_obses)
                old_val = q_network[agent].forward(s_obses).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
                loss = loss_fn(td_target, old_val)

                # Optimize the model for the critic
                optimizer[agent].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(q_network[agent].parameters()), config_args.max_grad_norm)
                optimizer[agent].step()


                # Actor training
                a, log_pi, action_probs = actor_network[agent].get_action(s_obses)

                # Compute the loss for this agent's actor
                # NOTE: Actor uses cross-entropy loss function where
                # input is the policy dist and the target is the value function with one-hot encoding applied
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
    :param csv_save_dir:
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
                    target = target_network[agent].forward(s_next_obses).gather(1, torch.LongTensor(actions_for_agent).view(-1,1)).squeeze().to(device)  # Size 32,1 
                    # Calculate the full TD target 
                    # NOTE that the target in this Fitted Q iteration implementation depends on the type of constraint we are using to 
                    # learn the policy
                    if (constraint == "queue"):
                        # Use the "g1" constraint
                        td_target = torch.Tensor(s_g1s).to(device) + config_args.gamma * target * (1 - torch.Tensor(s_dones).to(device))

                    elif (constraint == "speed_overage"):
                        # Use the "g2" constraint
                        td_target = torch.Tensor(s_g2s).to(device) + config_args.gamma * target * (1 - torch.Tensor(s_dones).to(device))

                    else: 
                        print("ERROR: Constraint function '{}' not recognized, unable to train using Fitted Q Iteration".format(constraint))

                # TODO: when calculating the "old value" should s_actions be used here? or should actions_for_agent? (i.e. should they come from
                # experience tuple or policy)
                old_val = q_network[agent].forward(s_obses).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()

                # print(f"target_network[agent].forward(s_next_obses) size: {target_network[agent].forward(s_next_obses).size()}")    # 32, 4
                # print(f"actions_for_agent size: {actions_for_agent.size()}")    # 32
                # print(f"target size: {target.size()}")  # 32, 1
                # print(f"td_target size: {td_target.size()}")    # 32, 32 (needs to be 32)
                # print(f"old_val size: {old_val.size()}")    # 32

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
        if (global_step % 100 == 0):

            # Evaluate the Q(s,a) of the first agent selecting the first action from the evaluation state
            eval_q_s_a = q_network[agents[0]].forward(eval_obses).squeeze()
            first_agent_first_action_value = eval_q_s_a[0].item()   # cast the tensor object to a float

            with open(f"{csv_save_dir}/FQE_loss.csv", "a", newline="") as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['Q(s,a) Sample', 'global_step'])
                csv_writer.writerow({**losses, **{'Q(s,a) Sample' : first_agent_first_action_value,
                                                            'global_step' : global_step}})

    stop_time = datetime.now()
    print(" >> Fitted Q Evaluation complete")
    print(" >>> Total execution time: {}".format(stop_time-start_time))
    
    return q_network


def EvaluatePolicyEnsemble(agent_ensembles:dict,
                            dataset:dict,
                            config_args) -> dict:
    """
    Each agent has a collection (i.e. "ensemble" of policies), in this function, we calculate the 
    expected value of the latest policy using the expected value of the previous policies.

    :param agent_ensembles: Dictionary that maps agents to lists of policies. The last policy in the list is the most
            recent policy
    :param dataset: Dictionary that maps agents to a collection of experience tuples
    :param config_args: Config arguments used to set up the experiment
    :returns a Dictionary that maps each agent to its expectation E_t[pi] = 1/t*(pi_t + (t-1)*E_t-1[pi])
    """

    print(f" > Evaluating Policy Expected Value")

    agents = agent_ensembles.keys()

    # Calculate policy outputs for each agent and each of its policy networks in the ensemble
    policy_outputs_ensemble = {agent : [] for agent in agents}
    expected_values_ensemble = {agent : [] for agent in agents}

    for agent in agents:
        
        agent_policy_outputs = []
        
        for policy in agent_ensembles[agent]:
            s_obses, s_actions, s_next_obses, s_g1s, s_g2s, s_dones = dataset[agent].sample(config_args.batch_size)
            action, log_prob, action_probs = policy.get_action(s_next_obses)    # TODO: do we use s_obses or s_next_obses here?
            
            agent_policy_outputs.append(action_probs)

        policy_outputs_ensemble[agent] = agent_policy_outputs

    # Calculate the expected value for each agent's policy using each agent's ensemble
    for agent in agents:
        # TODO: take mean here?
        agent_policy_outputs = policy_outputs_ensemble[agent]
        
        # print(f" >> agent: {agent} latest policy outputs: {agent_policy_outputs}")

        # Un-normalized sum
        cummulative_value = torch.zeros_like(agent_policy_outputs[0])
        
        # The number of policies (t) this agent has in the ensemble 
        num_policies = len(agent_policy_outputs)

        # Get the policy output of this agent's latest policy in the ensemble (pi_t)
        agent_latest_policy_output = policy_outputs_ensemble[agent][-1]

        for policy_output in agent_policy_outputs:
            cummulative_value += policy_output
        
        # Calculate 1/t*(pi + t-1(E[pi])) for this agent
        # TODO: we could also add weights to each policy in the agent's ensemble
        expected_value = 1/num_policies*(agent_latest_policy_output + (num_policies - 1)*cummulative_value)

        # The resulting expected_values_ensemble is a dictionary that maps each agent to be a tensor, 
        # where each tensor represents the expected action probabilities for that agent's policy 
        # according to using the ensemble of collected policies
        expected_values_ensemble[agent] = torch.mean(expected_value, dim=0)

    print(f" > Evaluation of policy expected value complete")

    return expected_values_ensemble


def EvaluateConstraintEnsemble(agent_ensembles:dict,
                            dataset:dict,
                            config_args) -> dict:
    """
    Each agent has a collection (i.e. "ensemble") of constraint value functions, in this function, we calculate the 
    expected value of the latest constraint value function using the expected value of the previous policies.

    :param agent_ensembles: Dictionary that maps agents to lists of policies. The last policy in the list is the most
            recent policy
    :param dataset: Dictionary that maps agents to a collection of experience tuples
    :param config_args: Config arguments used to set up the experiment
    :returns a Dictionary that maps each agent to its expectation E_t[g] = 1/t*(g_t + (t-1)*E_t-1[g])
    """
    print(f" > Evaluating Constraint Expected Value")

    agents = agent_ensembles.keys()

    # Calculate constraint value function outputs for each agent and each of its policy networks in the ensemble
    constraint_outputs_ensemble = {agent : [] for agent in agents}
    expected_values_ensemble = {agent : [] for agent in agents}

    for agent in agents:
        
        agent_constraint_outputs = []
        
        for constraint_function in agent_ensembles[agent]:
            s_obses, s_actions, s_next_obses, s_g1s, s_g2s, s_dones = dataset[agent].sample(config_args.batch_size)
            # TODO: this is the only line that is different from EvaluatePolicyEnsemble so these functions
            # should probbably be combined
            action_values = constraint_function.forward(s_obses)    
            
            agent_constraint_outputs.append(action_values)

        constraint_outputs_ensemble[agent] = agent_constraint_outputs


    # Calculate the expected value for each agent's constraint value function using each agent's ensemble
    for agent in agents:
        agent_constraint_outputs = constraint_outputs_ensemble[agent]
        
        # Un-normalized sum
        cummulative_value = torch.zeros_like(agent_constraint_outputs[0])
        
        # The number of policies (t) this agent has in the ensemble 
        num_policies = len(agent_constraint_outputs)

        # Get the policy output of this agent's latest policy in the ensemble (pi_t)
        agent_latest_constraint_output = constraint_outputs_ensemble[agent][-1]

        for constraint_output in agent_constraint_outputs:
            cummulative_value += constraint_output
        
        # Calculate 1/t*(g + t-1(E[g]))
        # TODO: we could also add weights to each policy in the agent's ensemble
        expected_value = 1/num_policies*(agent_latest_constraint_output + (num_policies - 1)*cummulative_value)

        # The resulting expected_values_ensemble is a dictionary that maps each agent to be a tensor, 
        # where each tensor represents the expected action values for that agent's constraint value function 
        # according to using the ensemble of collected constraint value functions
        expected_values_ensemble[agent] = torch.mean(expected_value, dim=0)

    print(f" > Evaluation of constraint expected value complete")

    return expected_values_ensemble


def PerformRollout(env:sumo_rl.parallel_env, policy:dict, config_args)->(dict, dict, dict):
    """
    Perform a 1-episode rollout of a provided policy to evaluate the constraint functions g1 and g2. 
    This function assumes that the environment has been set up with the 'queue' reward function when evaluating the
    g1 and g2 constraints.

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
    episode_rewards = {agent: 0 for agent in agents}            

    # Maps each agent to its MAX SPEED OVERAGE for this step        
    episode_constraint_1 = {agent : 0 for agent in agents}  
    
    # Maps each agent to the accumulated NUBMER OF CARS STOPPED for episode
    episode_constraint_2 = {agent : 0 for agent in agents}  

    # Initialize the env
    obses, _ = env.reset()

    # Perform the rollout
    for sumo_step in range(config_args.sumo_seconds):
        # Populate the action dictionary
        for agent in agents:
            # Only use optimal actions according to the policy
            action, _, _ = policy[agent].get_action(obses[agent])
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


# ---------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Parse the configuration file
    # Get config parameters                        
    parser = MARLConfigParser()
    args = parser.parse_args()

    if not args.seed:
        args.seed = int(datetime.now()) 

    # TODO: fix cuda...
    # device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    device = 'cpu'
    print(f"DEVICE: {device}")
    analysis_steps = args.analysis_steps                    # Defines which checkpoint will be loaded into the Q model
    parameter_sharing_model = args.parameter_sharing_model  # Flag indicating if we're loading a model from DQN with PS
    nn_load_directory = args.nn_directory 
    nn_dir = f"{nn_load_directory}"                              # Name of directory containing the stored nn from training
    
    # Initialize directories for logging, note that that models will be saved to subfolders created in the directory that was used to 
    # generate the dataset
    experiment_time = str(datetime.now()).split('.')[0].replace(':','-')   
    experiment_name = "{}__N{}__exp{}__seed{}__{}".format(args.gym_id, args.N, args.exp_name, args.seed, experiment_time)
    nn_save_dir = f"{nn_load_directory}/batch_offline_RL/{experiment_name}"
    csv_save_dir = f"{nn_save_dir}/csv" 
    os.makedirs(f"{nn_save_dir}/policies")
    os.makedirs(f"{nn_save_dir}/constraints")
    os.makedirs(csv_save_dir)

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

    if parameter_sharing_model:
        # Define the shape of the observation space depending on if we're using a global observation or not
        # Regardless, we need to add an array of length num_agents to the observation to account for one hot encoding
        eg_agent = agents[0]
        if args.global_obs:
            observation_space_shape = tuple((shape+1) * (num_agents) for shape in observation_spaces[eg_agent].shape)
        else:
            observation_space_shape = np.array(observation_spaces[eg_agent].shape).prod() + num_agents  # Convert (X,) shape from tuple to int so it can be modified
            observation_space_shape = tuple(np.array([observation_space_shape]))                        # Convert int to array and then to a tuple

        q_network = Actor(observation_space_shape, action_spaces[eg_agent].n, parameter_sharing_model).to(device) # In parameter sharing, all agents utilize the same q-network
        
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
            print(" > Loading NN from file: {} for dataset generation".format(nn_file))

    # Seed the env
    env.reset(seed=args.seed)
    for agent in agents:
        action_spaces[agent].seed(args.seed)
        observation_spaces[agent].seed(args.seed)

    # Initialize the env
    obses, _ = env.reset()


    """
    Step 1: 
        Generate the dataset using a previously trained model, the dataset should 
        contain information about the contstraint functions as well as the observations

        The dataset should also be generated using "optimal" actions (optimal according to the provided
        policy) ~80% of the time and the rest of the time random actions should be used
    """

    dataset = GenerateDataset(env, 
                              q_network, 
                              optimal_action_ratio=0.8, 
                              num_episodes=50,   
                              episode_steps=args.sumo_seconds)

    """
    Step 2:
        Use the generated dataset to iteratively learn a new policy 
        Essentially we want to evaluate the new policy, E[pi] and the constraint function E[G]
    """
    perform_rollout_comparisons = True
    policy_expectation, constraint_expectation = OfflineBatchRL(env,
                                                                dataset, 
                                                                q_network,
                                                                perform_rollout_comparisons,
                                                                args, 
                                                                nn_save_dir,
                                                                csv_save_dir,
                                                                constraint="queue")


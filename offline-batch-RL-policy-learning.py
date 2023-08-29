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

# SUMO dependencies
import sumo_rl
import sys
import os
from sumo_custom_observation import CustomObservationFunction
from sumo_custom_reward import MaxSpeedRewardFunction

# Config Parser
from MARLConfigParser import MARLConfigParser

# Custom modules 
from actor_critic import Actor, QNetwork
from dataset import Dataset
from linear_schedule import LinearSchedule


# Set up the system and environement
# Make sure SUMO env variable is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


def CalculateMaxSpeedOverage(max_speeds, speed_limit) -> :
    """
    Calculate how much the agents' max speeds exceeded some speed limit
    """
    agents = list(max_speeds.keys())
    agent_overages = {agent : 0 for agent in agents}

    for agent in agents:
        overage = 0.0
        agent_max_speed = max_speeds[agent]

        if agent_max_speed > speed_limit:
            overage = agent_max_speed - speed_limit

        agent_overages[agent] = overage


def GenerateDataset(env: sumo_rl.parallel_env, 
                    q_network: Actor, 
                    optimal_action_ratio:float = 0.8, 
                    num_episodes:int=100, 
                    episode_steps=1000) -> Dataset():
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
    print(">> Generating dataset")
    start_time = datetime.now()

    DATASET_SIZE = num_episodes*episode_steps
    SPEED_OVERAGE_THRESHOLD = 13.89
    agents = env.possible_agents
    
    # Initialize the dataset as a dictionary that maps agents to Dataset objects that are full of experience
    dataset = {agent : Dataset(DATASET_SIZE) for agent in agents}

    # Define empty dictionary tha maps agents to actions
    actions = {agent: None for agent in agents}
    action_spaces = env.action_spaces
    max_speeds = {agent: 0 for agent in agents}

    # Define dictionaries to hold the values of the constraints (g1 and g2) each step
    constraint_1 = {agent : 0 for agent in agents}  # Maps each agent to its MAX SPEED OVERAGE for this step
    constraint_2 = {agent : 0 for agent in agents}  # Maps each agent to the NUBMER OF CARS STOPPED for this step

    obses, _ = env.reset()

    for episode in num_episodes:

        for step in episode_steps:

            # Set the action for each agent
            for agent in agents:
                if random.random() < (1-optimal_action_ratio):
                    actions[agent] = action_spaces[agent].sample()
                else:
                    # Actor choses the actions
                    action, _, _ = q_network[agent].get_action(obses[agent])
                    actions[agent] = action.detach().cpu().numpy()

            # Apply all actions to the env
            next_obses, rewards, dones, truncated, info = env.step(actions)

            # Caclulate constraints and add the experience to the dataset
            for agent in agents:
                 max_speeds[agent] = next_obses[agent][-1]
                 constraint_1[agent] = CalculateMaxSpeedOverage(max_speeds, SPEED_OVERAGE_THRESHOLD)
                 constraint_2[agent] = rewards[agent]
                 dataset[agent].put((obses[agent], actions[agent], next_obses[agent], constraint_1[agent, constraint_2[agent]]))

    stop_time = datetime.now()
    print(">> Dataset generation complete")
    print(">>> Total execution time: {}".format(stop_time-start_time))
    return dataset


def OfflineBatchRL(observation_spaces:dict,
                   action_spaces:dict,
                   agents:list,
                   dataset: dict,
                   config_args,
                   constraint:str="") -> (dict, dict):

    # Check inputs
    if (constraint != "queue") or (constraint != "speed_overage")
        print("ERROR: Constraint function '{}' not recognized, unable to perform Offline Batch RL".format(constraint))

    # TODO: could make these configs
    MAX_NUM_ROUNDS = 20
    OMEGA = 0.1
    expectation_G2_prev = 0 # TODO: Initialize randomly

    for t in MAX_NUM_ROUNDS:

        # Learn a policy that optimizes actions for the "g2" constraint
        policies = FittedQIteration(observation_spaces, action_spaces, agents, dataset, config_args, constraint=constraint)  # TODO: Implement

        # Evaluate G_2^pi 
        G2_pi = FittedQEvaluation(policies, constraint=constraint)   # TODO: Implement

        # Update expectation for pi
        expectation_pi = 1/t * (policies + ((t-1)*expectation_pi))    # TODO: review this step

        # Update expectation for G2
        expectation_G2 = 1/t * (G2_pi + ((t-1)*expectation_G2_prev)) # TODO: review this step

        # Check exit condition
        if (np.linalg.norm(expectation_G2 - expectation_G2_prev)**2 <= OMEGA):
            break

    return expectation_pi, expectation_G2


def FittedQIteration(observation_spaces:dict,
                     action_spaces:dict,
                     agents:list,
                     dataset:dict, 
                     config_args, 
                     constraint:str="") -> dict:
    """
    Implementation of Fitted Q Iteration with function approximation for offline learning of a policy
    (algorithm 4 from Le, et. al)

    :param observation_spaces:
    :param action_spaces:
    :param agents:
    :param dataset:
    :param constraint:
    """


    print(">> Beginning Fitted Q Iteration")
    start_time = datetime.now()

    q_network = {}  # Dictionary for storing q-networks (maps agent to a q-network)
    target_network = {} # Dictionary for storing target networks (maps agent to a network)
    optimizer = {}  # Dictionary for storing optimizers for each RL problem
    
    for agent in agents:
        observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if config_args.global_obs else observation_spaces[agent].shape
        q_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
        target_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
        target_network[agent].load_state_dict(q_network[agent].state_dict())    # Intialize the target network the same as the main network
        optimizer[agent] = optim.Adam(q_network[agent].parameters(), lr=config_args.learning_rate) # All agents use the same optimizer for training

    # Define loss function as MSE loss
    loss_fn = nn.MSELoss() 

    # TODO: this should be updated to be for k = 1:K (does not need to be the same as total_timesteps)
    for global_step in range(config_args.total_timesteps):

        if config_args.global_obs:
            print("ERROR: global observations not supported for FittedQIteration")
            return {}

        # Training for each agent
        for agent in agents:

            # TODO: should this just happen every step in FQI? Or should  should we leave in global_step % args.train_frequency == 0
            if (global_step > config_args.learning_starts) and (global_step % config_args.train_frequency == 0):  
                
                # Sample data from the dataset
                s_obses, s_actions, s_next_obses, s_g1s, s_g2s = dataset[agent].sample(config_args.batch_size)
                
                # Compute the target
                with torch.no_grad():
                    # Calculate min_a Q(s',a)
                    target_min = torch.min(target_network[agent].forward(s_next_obses), dim=1)[0]
                    
                    # Calculate the full TD target 
                    # Note that the target in this Fitted Q iteration implementation depends on the type of constraint we are using to 
                    # learn the policy
                    if (constraint == "queue"):
                        # Use the "g1" constraint
                        td_target = torch.Tensor(s_g1s).to(device) + config_args.gamma * target_min 

                    elif (constraint == "speed_overage"):
                        # Use the "g2" constraint
                        td_target = torch.Tensor(s_g2s).to(device) + config_args.gamma * target_min 

                    else: 
                        print("ERROR: Constraint function '{}' not recognized, unable to train using Fitted Q Iteration".format(constraint))

                old_val = q_network[agent].forward(s_obses).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
                loss = loss_fn(td_target, old_val)

                # optimize the model
                optimizer[agent].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(q_network[agent].parameters()), config_args.max_grad_norm)
                optimizer[agent].step()

                # update the target network
                # TODO: should this just happen every time training occurs in FQI? 
                if global_step % args.target_network_frequency == 0:
                    target_network[agent].load_state_dict(q_network[agent].state_dict())

    stop_time = datetime.now()
    print(">> Fitted Q Iteration complete")
    print(">>> Total execution time: {}".format(stop_time-start_time))
    
    return q_network


def FittedQEvaluation(observation_spaces:dict,
                     action_spaces:dict,
                     agents:list,
                     policy:dict,
                     dataset:dict, 
                     config_args, 
                     constraint:str="") -> dict:
    """
    Implementation of Fitted Off-Policy Evaluation with function approximation for offline evaluation 
    of a policy
    (algorithm 3 from Le, et. al)

    :param observation_spaces:
    :param action_spaces:
    :param agents:
    :param policy:
    :param dataset:
    :param config_args:
    :param constraint:
    """


    print(">> Beginning Fitted Q Evaluation")
    start_time = datetime.now()

    q_network = {}  # Dictionary for storing q-networks (maps agent to a q-network)
    target_network = {} # Dictionary for storing target networks (maps agent to a network)
    optimizer = {}  # Dictionary for storing optimizers for each RL problem
    actions = {}

    for agent in agents:
        observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if config_args.global_obs else observation_spaces[agent].shape
        q_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
        target_network[agent] = QNetwork(observation_space_shape, action_spaces[agent].n).to(device)
        target_network[agent].load_state_dict(q_network[agent].state_dict())    # Intialize the target network the same as the main network
        optimizer[agent] = optim.Adam(q_network[agent].parameters(), lr=config_args.learning_rate) # All agents use the same optimizer for training
        actions[agent] = None

    # Define loss function as MSE loss
    loss_fn = nn.MSELoss() 

    # TODO: this should be updated to be for k = 1:K (does not need to be the same as total_timesteps)
    for global_step in range(config_args.total_timesteps):

        if config_args.global_obs:
            print("ERROR: global observations not supported for FittedQEvaluation")
            return {}

        # Training for each agent
        for agent in agents:

            if (global_step > config_args.learning_starts) and (global_step % config_args.train_frequency == 0):  
                
                # Sample data from the dataset
                s_obses, s_actions, s_next_obses, s_g1s, s_g2s = dataset[agent].sample(config_args.batch_size)
                
                # Use the sampled next observations (x') to generate actions according to the provided policy
                # NOTE this method of getting actions is identical to how it is performed in DQN
                logits = policy[agent].forward(s_obses.reshape((1,)+s_obses.shape))
                actions_for_agent = torch.argmax(logits, dim=1).tolist()[0]

                # Compute the target
                # NOTE That this is the only thing different between FQE and FQI
                with torch.no_grad():
                    # Calculate Q(s',pi(s'))
                    target = target_network[agent].forward(s_next_obses).gather(1, torch.LongTensor(actions_for_agent))

                    # Calculate the full TD target 
                    # NOTE that the target in this Fitted Q iteration implementation depends on the type of constraint we are using to 
                    # learn the policy
                    if (constraint == "queue"):
                        # Use the "g1" constraint
                        td_target = torch.Tensor(s_g1s).to(device) + config_args.gamma * target 

                    elif (constraint == "speed_overage"):
                        # Use the "g2" constraint
                        td_target = torch.Tensor(s_g2s).to(device) + config_args.gamma * target 

                    else: 
                        print("ERROR: Constraint function '{}' not recognized, unable to train using Fitted Q Iteration".format(constraint))

                # TODO: when calculating the "old value" should s_actions be used here? or should actions_for_agent? (i.e. should they come from
                # experience tuple or policy)
                old_val = q_network[agent].forward(s_obses).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
                loss = loss_fn(td_target, old_val)

                # optimize the model
                optimizer[agent].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(q_network[agent].parameters()), config_args.max_grad_norm)
                optimizer[agent].step()

                # Update the target network
                if global_step % args.target_network_frequency == 0:
                    target_network[agent].load_state_dict(q_network[agent].state_dict())

    stop_time = datetime.now()
    print(">> Fitted Q Evaluation complete")
    print(">>> Total execution time: {}".format(stop_time-start_time))
    
    return q_network


# ---------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Parse the configuration file
    # Get config parameters                        
    parser = MARLConfigParser()
    args = parser.parse_args()

    if not args.seed:
        args.seed = int(datetime.now()) 

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    analysis_steps = args.analysis_steps                    # Defines which checkpoint will be loaded into the Q model
    parameter_sharing_model = args.parameter_sharing_model  # Flag indicating if we're loading a model from DQN with PS
    nn_directory = args.nn_directory 
    nn_dir = f"{nn_directory}"                              # Name of directory containing the stored nn from training

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
    # TODO: these dictionaries are deprecated, use action_space & observation_space functions instead
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
            print("> Loading NN from file: {} for dataset generation".format(nn_file))

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
                              num_episodes=100, 
                              episode_steps=args.sumo_seconds)

    """
    Step 2:
        Use the generated dataset to learn a new policy 
        Essentially we want to evaluate the new policy, E[pi] and the constraint function E[G]
    """
    policy_expectation, constraint_expectation = OfflineBatchRL(env, dataset, args, constraint="queue")


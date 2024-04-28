from datetime import datetime
import os
import sys
import random
import csv
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import sumo_rl

from MARLConfigParser import MARLConfigParser

from rl_core.actor_critic import Actor, QNetwork, one_hot_q_values
from rl_core.fitted_q_evaluation import FittedQEvaluation
from rl_core.rollout import OfflineRollout, OnlineRollout

from marl_utils.dataset import GenerateDataset
from sumo_custom_observation import CustomObservationFunction


# Make sure SUMO env variable is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")



def AblationStudy(env:sumo_rl.parallel_env,
                dataset: dict,
                dataset_policies:list,
                config_args,
                nn_save_dir:str,
                csv_save_dir:str,
                device:torch.device='cpu') -> tuple[dict, dict, list, list]:

    """
    Perform the offline batch learning ablation study
    The goal here is to compare the G1 and G2 value functions learned by FQE with what the "true" values are of the 
    environment. So the G1 and G2 constraint value functions will each be learned using both dataset policies and 
    then compared to the online returns using the those same policies

    :param env: The SUMO environment that was used to generate the dataset
    :param dataset: Dictionary that maps each agent to its experience tuple
    :param dataset_policies: List of policies that were used to generate the dataset for this experiment (used for online evaluation)
    :param config_args: Configuration arguments used to set up the experiment
    :param nn_save_dir: Directory in which to save the models each round
    :pram csv_save_dir: Directory in which to save the csv file
    :param device: Torch device with which to do the reinforcement learning
    """


    print(f" > Starting batch policy learning ablation study...")
    function_start_time = datetime.now()

    # NOTE: These are the raw objects from the environment, they should not be modified for parameter sharing
    # Dimensions of the observation space can be modified within sub-functions where necessary
    agents = env.possible_agents
    action_spaces = env.action_spaces
    observation_spaces = env.observation_spaces

    # Initialize csv files
    with open(f"{csv_save_dir}/offline_rollouts.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=[agent + '_g1_threshold_policy_threshold_return' for agent in agents] +
                                                        ['total_g1_threshold_policy_threshold_return'] +
                                                        [agent + '_g2_threshold_policy_threshold_return' for agent in agents] +
                                                        ['total_g2_threshold_policy_threshold_return'] +

                                                        [agent + '_g1_queue_policy_queue_return' for agent in agents] +
                                                        ['total_g1_queue_policy_queue_return'] +
                                                        [agent + '_g2_queue_policy_queue_return' for agent in agents] +
                                                        ['total_g2_queue_policy_queue_return'])
        csv_writer.writeheader()

    with open(f"{csv_save_dir}/online_rollouts.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=[agent + '_threshold_policy_g1_return' for agent in agents] +
                                                        ['threshold_policy_system_return_g1'] +
                                                        [agent + '_threshold_policy_g2_return' for agent in agents] +
                                                        ['threshold_policy_system_return_g2'] +

                                                        [agent + '_queue_policy_g1_return' for agent in agents] +
                                                        ['queue_policy_system_return_g1'] +
                                                        [agent + '_queue_policy_g2_return' for agent in agents] +
                                                        ['queue_policy_system_return_g2'])
        csv_writer.writeheader()

    # Define a "mini" dataset to be used for offline rollouts (basically this will get passed through networks to evaluate them)
    rollout_mini_dataset = {}
    for agent in agents:

        # TODO: config, currently set to the same size as the dataset itself
        sample_size = len(dataset[agent].buffer)
        print(f"   > Generating mini dataset of size: {sample_size}")
        rollout_mini_dataset[agent] = dataset[agent].sample(sample_size)


    # Learn G1 value function according to speed threshold policy
    print(f"   > Learning G1_pi using 'speed threshold' policy (G1_threshold_policy)")
    G1_threshold_policy = FittedQEvaluation(observation_spaces=observation_spaces, 
                                            action_spaces=action_spaces, 
                                            agents=agents,
                                            policies=dataset_policies[0],    # Assumes order was [threshold, queue]
                                            dataset=dataset,
                                            csv_save_dir=csv_save_dir,
                                            csv_file_suffix='g1_threshold',    
                                            config_args=config_args, 
                                            constraint="average-speed-limit")

    # Learn G2 value function according to speed threshold policy
    print(f"   > Learning G2_pi using 'speed threshold' policy (G2_threshold_policy)")    
    G2_threshold_policy = FittedQEvaluation(observation_spaces=observation_spaces, 
                                            action_spaces=action_spaces, 
                                            agents=agents,
                                            policies=dataset_policies[0],    # Assumes order was [threshold, queue]
                                            dataset=dataset,
                                            csv_save_dir=csv_save_dir,
                                            csv_file_suffix='g2_threshold',    
                                            config_args=config_args, 
                                            constraint="queue")
    
    # Learn G1 value function according to queue policy
    print(f"   > Learning G1_pi using 'queue' policy (G1_queue_policy)")
    G1_queue_policy = FittedQEvaluation(observation_spaces=observation_spaces, 
                                        action_spaces=action_spaces, 
                                        agents=agents,
                                        policies=dataset_policies[1],    # Assumes order was [threshold, queue]
                                        dataset=dataset,
                                        csv_save_dir=csv_save_dir,
                                        csv_file_suffix='g1_queue',    
                                        config_args=config_args, 
                                        constraint="average-speed-limit")
    
    # Learn G2 value function according to queue policy
    print(f"   > Learning G2_pi using 'queue' policy (G2_queue_policy)")
    G2_queue_policy = FittedQEvaluation(observation_spaces=observation_spaces, 
                                        action_spaces=action_spaces, 
                                        agents=agents,
                                        policies=dataset_policies[1],    # Assumes order was [threshold, queue]
                                        dataset=dataset,
                                        csv_save_dir=csv_save_dir,
                                        csv_file_suffix='g2_queue',    
                                        config_args=config_args, 
                                        constraint="queue")
    
    # Perform an offline rollout using G1_threshold_policy and speed threshold policy
    print(f"   > Evaluating G1_threshold_policy in offline rollout using 'speed threshold' policy")
    offline_g1_returns_threshold_policy = OfflineRollout(value_function=G1_threshold_policy, 
                                                        policies=dataset_policies[0], 
                                                        mini_dataset=rollout_mini_dataset,  # TODO: Update with dataset that only has threshold observations
                                                        device=device)

    # Perform an offline rollout using G2_speed_threshold and speed threshold policy
    print(f"   > Evaluating G2_threshold_policy in offline rollout using 'speed threshold' policy")
    offline_g2_returns_threshold_policy = OfflineRollout(value_function=G2_threshold_policy, 
                                                        policies=dataset_policies[0], 
                                                        mini_dataset=rollout_mini_dataset,  # TODO: Update with dataset that only has threshold observations
                                                        device=device)
    
    # Perform an offline rollout using G1_queue_policy and queue policy
    print(f"   > Evaluating G1_queue_policy in offline rollout using 'queue' policy")
    offline_g1_returns_queue_policy = OfflineRollout(value_function=G1_queue_policy, 
                                                    policies=dataset_policies[1], 
                                                    mini_dataset=rollout_mini_dataset,  # TODO: Update with dataset that only has queue observations
                                                    device=device)
    
    # Perform an offline rollout using G2_queue and queue policy
    print(f"   > Evaluating G2_queue_policy in offline rollout using 'queue' policy")
    offline_g2_returns_queue_policy = OfflineRollout(value_function=G2_queue_policy, 
                                                    policies=dataset_policies[1], 
                                                    mini_dataset=rollout_mini_dataset,  # TODO: Update with dataset that only has queue observations
                                                    device=device)
    
    # Perform an online rollout using speed threshold policy
    print(f"   > Performing online rollout using 'speed threshold' policy")
    _, g1_online_return_threshold, g2_online_return_threshold = OnlineRollout(env=env, 
                                                                            policies=dataset_policies[0],
                                                                            config_args=config_args,
                                                                            device=device)

    # Perform an online rollout using queue policy 
    print(f"   > Performing online rollout using 'queue' policy")
    _, g1_online_return_queue, g2_online_return_queue = OnlineRollout(env=env, 
                                                                    policies=dataset_policies[1],
                                                                    config_args=config_args,
                                                                    device=device)    


    # Save the offline and online returns to CSV so they can be compared
    with open(f"{csv_save_dir}/offline_rollouts.csv", "a", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=[agent + '_g1_threshold_policy_threshold_return' for agent in agents] +
                                                        ['total_g1_threshold_policy_threshold_return'] +
                                                        [agent + '_g2_threshold_policy_threshold_return' for agent in agents] +
                                                        ['total_g2_threshold_policy_threshold_return'] +

                                                        [agent + '_g1_queue_policy_queue_return' for agent in agents] +
                                                        ['total_g1_queue_policy_queue_return'] +
                                                        [agent + '_g2_queue_policy_queue_return' for agent in agents] +
                                                        ['total_g2_queue_policy_queue_return'])
        new_row = {}

        for agent in agents:
            new_row[agent + '_g1_threshold_policy_threshold_return'] = offline_g1_returns_threshold_policy[agent].item()
            new_row[agent + '_g2_threshold_policy_threshold_return'] = offline_g2_returns_threshold_policy[agent].item()
            new_row[agent + '_g1_queue_policy_queue_return'] = offline_g1_returns_queue_policy[agent].item()
            new_row[agent + '_g2_queue_policy_queue_return'] = offline_g2_returns_queue_policy[agent].item()

        new_row['total_g1_threshold_policy_threshold_return'] = torch.sum(torch.tensor(list(offline_g1_returns_threshold_policy.values()))).detach().numpy()
        new_row['total_g2_threshold_policy_threshold_return'] = torch.sum(torch.tensor(list(offline_g2_returns_threshold_policy.values()))).detach().numpy()
        new_row['total_g1_queue_policy_queue_return'] = torch.sum(torch.tensor(list(offline_g1_returns_queue_policy.values()))).detach().numpy()
        new_row['total_g2_queue_policy_queue_return'] = torch.sum(torch.tensor(list(offline_g2_returns_queue_policy.values()))).detach().numpy()

        csv_writer.writerow({**new_row})

    with open(f"{csv_save_dir}/online_rollouts.csv", "a", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=[agent + '_threshold_policy_g1_return' for agent in agents] +
                                                        ['threshold_policy_system_return_g1'] +
                                                        [agent + '_threshold_policy_g2_return' for agent in agents] +
                                                        ['threshold_policy_system_return_g2'] +

                                                        [agent + '_queue_policy_g1_return' for agent in agents] +
                                                        ['queue_policy_system_return_g1'] +
                                                        [agent + '_queue_policy_g2_return' for agent in agents] +
                                                        ['queue_policy_system_return_g2'])
        new_row = {}
        for agent in agents:

            new_row[agent + '_threshold_policy_g1_return'] = g1_online_return_threshold[agent]
            new_row[agent + '_threshold_policy_g2_return'] = g2_online_return_threshold[agent]

            new_row[agent + '_queue_policy_g1_return'] = g1_online_return_queue[agent]
            new_row[agent + '_queue_policy_g2_return'] = g2_online_return_queue[agent]


        new_row['threshold_policy_system_return_g1'] = sum(g1_online_return_threshold.values())
        new_row['threshold_policy_system_return_g2'] = sum(g2_online_return_threshold.values())

        new_row['queue_policy_system_return_g1'] = sum(g1_online_return_queue.values())
        new_row['queue_policy_system_return_g2'] = sum(g2_online_return_queue.values())

        csv_writer.writerow({**new_row})


    function_stop_time = datetime.now()
    print(f" > Batch offline ablation study")
    print(f" > Total execution time: {function_stop_time-function_start_time}")

    return 

# ---------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    print(f" > Setting up for batch offline ablation study")

    # Parse the configuration file for experiment configuration parameters
    parser = MARLConfigParser()
    args = parser.parse_args()

    if not args.seed:
        args.seed = int(datetime.now())

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f" > DEVICE: {device}")

    # Defines which checkpoint will be loaded into the Q model
    analysis_steps = args.analysis_steps                            
    
    # Flag indicating if we're loading a model from DQN with PS
    parameter_sharing_model = args.parameter_sharing_model
    
    # Name of directory containing the stored queue model nn from training
    nn_queue_dir = f"{args.nn_queue_directory}"

    # Name of directory containing the stored speed overage model nn from training
    nn_avg_speed_limit_dir = f"{args.nn_speed_overage_directory}"   

    # Initialize directories for logging, note that that models will be saved to subfolders created in the directory that was used to 
    # generate the dataset
    experiment_time = str(datetime.now()).split('.')[0].replace(':','-')
    experiment_name = "{}__N{}__exp{}__seed{}__{}".format(args.gym_id, args.N, args.exp_name, args.seed, experiment_time)
    save_dir = f"ablation_study/{experiment_name}"
    csv_save_dir = f"{save_dir}/csv"
    os.makedirs(csv_save_dir)

    # Create the env
    # Sumo must be created using the sumo-rl module
    # NOTE: we have to use the parallel env here to conform to this implementation of dqn
    if (args.sumo_reward != "queue"):
        print(f"  > WARNING: Reward '{args.sumo_reward}' specified but being ignored")
    print(f"    > Setting up environment with standard 'queue' reward for ablation study")
    print(f"    > This is to ensure that the 'g1' constraint always corresponds to speed threshold and the 'g2' "+\
          f"constraint corresponds to queue length")
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

    print("\n=================== Environment Information ===================")
    print(" > agents: {}".format(agents))
    print(" > num_agents: {}".format(num_agents))
    print(" > action_spaces: {}".format(action_spaces))
    print(" > observation_spaces: {}".format(observation_spaces))

    # Construct the Q-Network model
    # Note the dimensions of the model varies depending on if the parameter sharing algorithm was used or the normal independent
    # DQN model was used
    list_of_policies = []
    eg_agent = agents[0]
    queue_model_policies = {}  # Dictionary for storing q-networks (maps agent to a q-network)
    avg_speed_limit_model_policies = {}    

    print("  > Parameter Sharing Enabled: {}".format(parameter_sharing_model))

    if parameter_sharing_model:
        # In this case, each agent will still have its own network, they will just be copies of each other

        # Define the shape of the observation space depending on if we're using a global observation or not
        # Regardless, we need to add an array of length num_agents to the observation to account for one hot encoding
        if args.global_obs:
            observation_space_shape = tuple((shape+1) * (num_agents) for shape in observation_spaces[eg_agent].shape)
        else:
            # Convert (X,) shape from tuple to int so it can be modified
            observation_space_shape = np.array(observation_spaces[eg_agent].shape).prod() + num_agents
            # Convert int to array and then to a tuple
            observation_space_shape = tuple(np.array([observation_space_shape]))                        

        # Queue model policies, In parameter sharing, all agents utilize the same q-network
        queue_model_policy = Actor(observation_space_shape, action_spaces[eg_agent].n).to(device) 
        
        # Load the Q-network file
        nn_queue_file = "{}/{}.pt".format(nn_queue_dir, analysis_steps)
        queue_model_policy.load_state_dict(torch.load(nn_queue_file))
        print(" > Loading NN from file: {} for 'queue' policy".format(nn_queue_file))

        # Speed overage model policies, In parameter sharing, all agents utilize the same q-network
        avg_speed_limit_model_policy = Actor(observation_space_shape, action_spaces[eg_agent].n).to(device) 

        # Load the Q-network file
        nn_avg_speed_limit_file = "{}/{}.pt".format(nn_avg_speed_limit_dir, analysis_steps)
        avg_speed_limit_model_policy.load_state_dict(torch.load(nn_avg_speed_limit_file))
        print(" > Loading NN from file: {} for 'average speed limit' policy".format(nn_avg_speed_limit_file))

        for agent in agents: 
            avg_speed_limit_model_policies[agent] = avg_speed_limit_model_policy
            queue_model_policies[agent] = queue_model_policy


    # Else the agents were trained using normal independent DQN so each agent gets its own Q-network model
    else: 
        
        # Load the Q-Network NN model for each agent from the specified anaylisis checkpoint step from training
        for agent in agents:
            observation_space_shape = tuple(shape * num_agents for shape in observation_spaces[agent].shape) if args.global_obs else observation_spaces[agent].shape
            queue_model_policies[agent] = Actor(observation_space_shape, action_spaces[agent].n)
            avg_speed_limit_model_policies[agent] = Actor(observation_space_shape, action_spaces[agent].n)

            # Queue model policies
            nn_queue_file = "{}/{}-{}.pt".format(nn_queue_dir, analysis_steps, agent)
            queue_model_policies[agent].load_state_dict(torch.load(nn_queue_file))
            print(" > Loading NN from file: {} for 'queue' policy".format(nn_queue_file))

            # Repeat for speed overage model policies
            nn_avg_speed_limit_file = "{}/{}-{}.pt".format(nn_avg_speed_limit_dir, analysis_steps, agent)
            avg_speed_limit_model_policies[agent].load_state_dict(torch.load(nn_avg_speed_limit_file))
            print(" > Loading NN from file: {} for 'average speed limit' policy".format(nn_avg_speed_limit_file))

    # List of policies is [avg_speed_limit, queue] - the order is imporant
    list_of_policies.append(avg_speed_limit_model_policies)
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
        Generate an un-normalized dataset (or load one)
    """
    if (args.dataset_path == ""):
        # No dataset provided so need to generate one

        # Path to save the dataset
        dataset_save_dir = f"{save_dir}/dataset"
        os.makedirs(dataset_save_dir)

        # We need to generate the dataset
        # TODO: add these arguments to config file
        dataset = GenerateDataset(env, 
                              list_of_policies, 
                              avg_speed_action_ratio=0.4,
                              queue_action_ratio=0.4, 
                              num_episodes=50,   
                              episode_steps=args.sumo_seconds,
                              parameter_sharing_model=args.parameter_sharing_model,
                              device=device)
        
        with open(f"{dataset_save_dir}/dataset.pkl", "wb") as f:
            pickle.dump(dataset, f)

    else:
        print(f" > Loading dataset from: {args.dataset_path}")
        # Load the previously generated dataset
        with open(f"{args.dataset_path}", "rb") as f:
            dataset = pickle.load(f)

    print(f" > Dataset size: {len(dataset[eg_agent].buffer)} per agent")

    """
    Step 2: 
        Run ablation study
    """
    AblationStudy(env=env,
                dataset=dataset,
                dataset_policies=list_of_policies,
                config_args=args,
                nn_save_dir='', # TODO: add logging for the learned FQE value functions
                csv_save_dir=csv_save_dir,
                device=device)
    
    # All done
    sys.exit(0)
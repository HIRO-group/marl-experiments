
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from datetime import datetime
import random
import csv
import sys

from rl_core.actor_critic import QNetwork


def FittedQEvaluation(observation_spaces:dict,
                     action_spaces:dict,
                     agents:list,
                     policies:dict,
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
    :param policies: Dictionary that maps an agent to its policy to be evaluated
    :param dataset: Dictionary that maps each agent to its experience tuple
    :param csv_save_dir: Path to the directory being used to store CSV files for this experiment
    :param csv_file_suffix: String to append to the name of the csv file to differentiate between which mean constraint value function is being evaluated
    :param config_args: Configuration arguments used to set up the experiment
    :param constraint: 'average-speed-limit' or 'queue', defines how the target should be determined while learning the value function
    :returns A dictionary that maps each agent to its learned constraint value function
    """



    print("     > Beginning Fitted Q Evaluation")
    start_time = datetime.now()

    device = torch.device('cuda' if torch.cuda.is_available() and config_args.cuda else 'cpu')
    print(f"      > FittedQEvaluation device: {device}")

    with open(f"{csv_save_dir}/FQE_loss_{csv_file_suffix}.csv", "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=agents + ['Q(s,a) Sample','global_step'])
        csv_writer.writeheader()

    # TODO: implement
    if config_args.global_obs:
        print("ERROR: global observations not supported for FittedQEvaluation")
        return {}

    if config_args.parameter_sharing_model:
        # Parameter sharing is used so there is only a single actor and critic network
        print(f"     > Parameter sharing enabled")
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
        print(f"    > Parameter sharing NOT enabled")

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
                actions_for_agent, _, _ = policies[random_agent].get_action(s_obses)

                # Compute the target
                # NOTE That this is the only thing different between FQE and FQI
                with torch.no_grad():
                    
                    # Calculate Q(s',pi(s'))
                    target = target_network.forward(s_next_obses).gather(1, torch.Tensor(actions_for_agent).view(-1,1).to(device)).squeeze()
                    
                    # Calculate the full TD target 
                    # NOTE that the target in this Fitted Q evaluation implementation depends on the type of constraint we are using to 
                    # learn the policy
                    if (constraint == "average-speed-limit"):
                        # Use the "g1" constraint
                        td_target = torch.Tensor(s_g1s).to(device) + config_args.gamma * target * (1 - torch.Tensor(s_dones).to(device))

                    elif (constraint == "queue"):
                        # Use the "g2" constraint
                        td_target = torch.Tensor(s_g2s).to(device) + config_args.gamma * target * (1 - torch.Tensor(s_dones).to(device))

                    else: 
                        print("ERROR: Constraint function '{}' not recognized, unable to train using Fitted Q Evaluation".format(constraint))
                        sys.exit(1)

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
                    actions_for_agent, _, _ = policies[agent].get_action(s_obses)
                    
                    # Compute the target
                    # NOTE That this is the only thing different between FQE and FQI
                    with torch.no_grad():
                        
                        # Calculate Q(s',pi(s'))
                        target = target_network[agent].forward(s_next_obses).gather(1, torch.Tensor(actions_for_agent).view(-1,1).to(device)).squeeze()
                        
                        # Calculate the full TD target 
                        # NOTE that the target in this Fitted Q evaluation implementation depends on the type of constraint we are using to 
                        # learn the policy
                        if (constraint == "average-speed-limit"):
                            # Use the "g1" constraint
                            td_target = torch.Tensor(s_g1s).to(device) + config_args.gamma * target * (1 - torch.Tensor(s_dones).to(device))

                        elif (constraint == "queue"):
                            # Use the "g2" constraint
                            td_target = torch.Tensor(s_g2s).to(device) + config_args.gamma * target * (1 - torch.Tensor(s_dones).to(device))

                        else: 
                            print("ERROR: Constraint function '{}' not recognized, unable to train using Fitted Q Evaluation".format(constraint))
                            sys.exit(1)

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
                    if global_step % config_args.target_network_frequency == 0:
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
    print("     > Fitted Q Evaluation complete")
    print("       > Total execution time: {}".format(stop_time-start_time))

    # If we're using parameter sharing, there is only a single network so to conform to the rest of the code, return a dictionary that maps
    # each agent to it's constraint network
    if config_args.parameter_sharing_model:
        q_network = {agent: q_network for agent in agents}

    return q_network
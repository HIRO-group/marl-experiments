
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from datetime import datetime
import random
import csv
import sys

from rl_core.actor_critic import Actor, QNetwork, one_hot_q_values


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
    :param constraint: 'average-speed-limit', 'queue', or 'weighted_sum' defines how the target should be determined while learning the policy
    :param lambda_1: Weight corresponidng to the G1 constraints, if constraint is 'weighted_sum' this must be provided
    :param lambda_2: Weight corresponidng to the G2 constraints, if constraint is 'weighted_sum' this must be provided
    :returns A dictionary that maps each agent to its learned policy
    """

    print(" > Beginning Fitted Q Iteration")
    start_time = datetime.now()

    device = torch.device('cuda' if torch.cuda.is_available() and config_args.cuda else 'cpu')
    print(f" > FittedQIteration DEVICE: {device}")

    if config_args.global_obs:
        print("ERROR: global observations not supported for FittedQIteration")
        return {}

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
                    if (constraint == "average-speed-limit"):
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
                        sys.exit(1)

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
                        if (constraint == "average-speed-limit"):
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
                            sys.exit(1)

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
    print(" > Fitted Q Iteration complete")
    print("   > Total execution time: {}".format(stop_time-start_time))
    
    # If we're using parameter sharing, there is only a single network so to conform to the rest of the code, return a dictionary that maps
    # each agent to it's policy network
    if config_args.parameter_sharing_model:
        actor_network = {agent: actor_network for agent in agents}

    return actor_network
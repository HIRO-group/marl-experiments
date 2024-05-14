
import collections
import torch
import random
import numpy as np
from datetime import datetime


from calculate_speed_control import CalculateSpeedError, CalculateMaxSpeedPension

# Based on ReplayBuffer class 
# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class Dataset():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, s_prime_lst, g1_list, g2_list, dones_list = [], [], [], [], [], []
        
        for transition in mini_batch:
            s, a, s_prime, g1, g2, dones = transition
            s_lst.append(s)
            a_lst.append(a)
            s_prime_lst.append(s_prime)
            g1_list.append(g1)
            g2_list.append(g2)
            dones_list.append(dones)

        return np.array(s_lst), np.array(a_lst), \
               np.array(s_prime_lst), np.array(g1_list), \
               np.array(g2_list), np.array(dones_list)
    

def GenerateDataset(env,
                    list_of_policies: list,
                    avg_speed_action_ratio:float = 0.4,
                    queue_action_ratio:float = 0.4, 
                    num_episodes:int=100, 
                    episode_steps:int=1000,
                    parameter_sharing_model:bool=False,
                    device:torch.device='cpu') -> dict:
    """
    :param env: The environment
    :param list_of_policies: A list of trained neural networks used to generate the dataset by acting in the environment
        NOTE: This function assumes that the list of policies is ordered like [avg_speed_model, queue_model]
    :param avg_speed_action_ratio: The fraction of actions that should come from the queue policy when generating  
            the dataset, if avg_speed_action_ratio + queue_action_ratio are less than 1, the remaining fraction of actions
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
    print(f"  > Generating dataset with {num_episodes} episodes of {episode_steps} steps")
    print(f"    > Optimal 'queue' action ratio: {queue_action_ratio}")
    print(f"    > Optimal 'avg speed limit' action ratio: {avg_speed_action_ratio}")
    start_time = datetime.now()

    DATASET_SIZE = num_episodes*episode_steps

    SPEED_LIMIT = 7.0
    
    agents = env.possible_agents
    num_agents = len(agents)
    optimal_action_ratio = avg_speed_action_ratio + queue_action_ratio  # Ratio of optimal actions that should be in this dataset

    onehot_keys = {agent: i for i, agent in enumerate(agents)}

    # Initialize the dataset as a dictionary that maps agents to Dataset objects that are full of experience
    dataset = {agent : Dataset(DATASET_SIZE) for agent in agents}

    # Define empty dictionary tha maps agents to actions
    actions = {agent: None for agent in agents}
    action_spaces = env.action_spaces

    # Define dictionaries to hold the values of the constraints (g1 and g2) each step
    constraint_1 = {agent : 0 for agent in agents}  # Maps each agent to its (-1) * AVG SPEED ERROR for this step
    constraint_2 = {agent : 0 for agent in agents}  # Maps each agent to the (-1) * NUBMER OF CARS STOPPED for this step

    for episode in range(num_episodes):
        print(f"    > Generating episode: {episode}")

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
                q_network = np.random.choice(['random'] + list_of_policies, 1, p=[1-optimal_action_ratio, avg_speed_action_ratio, queue_action_ratio])[0]

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
                print(f"    > Episode complete at {step} steps, going to next episode")
                break

            if parameter_sharing_model:
                for agent in agents:
                    onehot = np.zeros(num_agents)
                    onehot[onehot_keys[agent]] = 1.0
                    next_obses[agent] = np.hstack([onehot, next_obses[agent]])

            # Caclulate constraints and add the experience to the dataset
            for agent in agents:
                max_speed_observed_by_agent = next_obses[agent][-1]
                avg_speed_observed_by_agent = next_obses[agent][-2]

                # TODO: Make constraint definitions configurable
                # constraint_1[agent] = CalculateMaxSpeedPension(speed=max_speed_observed_by_agent) 
                constraint_1[agent] = CalculateSpeedError(speed=avg_speed_observed_by_agent, 
                                                        speed_limit=SPEED_LIMIT,
                                                        lower_speed_limit=SPEED_LIMIT)
                constraint_2[agent] = rewards[agent]   # NOTE: This assumes that the environment was configured with the "queue" reward
                dataset[agent].put((obses[agent], actions[agent], next_obses[agent], constraint_1[agent], constraint_2[agent], dones[agent]))

            obses = next_obses


    stop_time = datetime.now()
    print("  > Dataset generation complete")
    print("    > Total execution time: {}".format(stop_time-start_time))

    env.close()

    return dataset    
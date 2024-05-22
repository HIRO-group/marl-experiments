import torch
import numpy as np

from calculate_speed_control import CalculateSpeedError, CalculateMaxSpeedPension


def OfflineRollout(value_function:dict, policies:dict, mini_dataset:dict, device:torch.device)->dict:
    """
    :param value_function: Dictionary that maps agents to the learned value function for a given constraint
    :param policies: Dictionary that maps agents to the policy that should be used to select actions from the mini_dataset for evaluation
    :param mini_dataset: A subset of the larger dataset that contains experience tuples that are evaluated in the rollout
            using the provided policy and value function
            NOTE: This function assumes the dataset has been sampled such that mini_dataset contains lists of of experience
            tuples for each agent
    :param device: The pytorch device to use for tensor math
    :returns Dictionary containing the cummulative return for all agents of the mini dataset according to the provided value functions
            normalized by the size of the mini dataset
    """

    agents = list(value_function.keys())
    cumulative_return = {agent:0.0 for agent in agents}

    for agent in agents:
        
        # NOTE: when using parameter sharing, the observations here should already have 
        # one hot encoding applied
        # if (agent == '1'): print(f"     >>> mini_dataset: {mini_dataset[agent]}\n")
        obses_array, actions_array, next_obses_array, _, _, _ = mini_dataset[agent] # TODO: should I use obses or next_obses??
        # if (agent == '1'): print(f"       >>> obses_array: {obses_array}\n")

        # Get the max_a Q(s,a) for the observation
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TODO: POLICY IS NOT ALWAYS PRODUCING THE OPTIMAL ACITON!!!!!!!!!!!!!!!!!!!!!!!!!!!
        actions_from_policy, _, _ = policies[agent].to(device).get_action(obses_array)  
        # if (agent == '1'): print(f"     >>> actions_from_policy: {actions_from_policy}\n")

        # Evaluate the actions that were selected by the policies from this observation
        # if (agent == '1'): print(f"     >>> values (BEFORE GATHER): {value_function[agent].forward(obses_array).to(device)}\n")
        values = (value_function[agent].forward(obses_array)).to(device).gather(1, actions_from_policy.view(-1,1)).squeeze()
        # if (agent == '1'): print(f"     >>> values (AFTER GATHER): {values}\n")

        # Add up the values of all states from the mini dataset and normalize it by the size of the dataset
        print(f"     > Normalizing offline rollout return for agent '{agent}' by size of mini dataset: {len(obses_array)}")
        cumulative_return[agent] = sum(values)/(len(obses_array))

    return cumulative_return





def OnlineRollout(env, policies:dict, config_args, device:torch.device)->tuple[dict, dict, dict]:
    """
    Perform a 1-episode rollout of a provided policy to evaluate the constraint functions g1 and g2.
    This function assumes that the environment has been set up with the 'queue' reward function when evaluating the
    g1 and g2 constraints.

    NOTE: this is an "online" rollout because it assumes that all agents are using their current learned policy

    :param env: The environment to execute the policy in
    :param policies: Dictionary that maps agents to "actor" models
    :param config_args: Configuration arguments used to set up the experiment
    :returns: Three dictionaries, the first dict maps agents to their accumulated reward for the episode,
            the second dict maps agents to their accumulated g1 constraint for the episode, the third dict
            maps agents to their accumulated g2 constraint for the episode
    """
    # TODO: update function to support global observations

    # Define the speed limit used to evaluate the g1 constraint
    # (note this needs to match what is used in GenerateDataset)
    SPEED_LIMIT = 7.0 

    agents = env.possible_agents
    num_agents = len(agents)

    # This function assumes that the environment is set up with the 'queue' reward so if that's not the case 
    # just remind the user
    if (config_args.sumo_reward != 'queue'):
        print(f"  > WARNING: Reward '{config_args.sumo_reward}' specified but being ignored\n" \
              f"  > Online rollout being performed with 'queue' reward to match 'g1' and 'g2' definitions.")

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
            action, _, _ = policies[agent].to(device).get_action(obses[agent])
            actions[agent] = action.detach().cpu().numpy()

        # Apply all actions to the env
        next_obses, rewards, dones, truncated, info = env.step(actions)

        if np.prod(list(dones.values())):
            print(f" > Episode complete at after {sumo_step} steps")                
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
            episode_rewards[agent] += rewards[agent]
            max_speed_observed_by_agent = next_obses[agent][-1]
            avg_speed_observed_by_agent = next_obses[agent][-2]
            # episode_constraint_1[agent] += CalculateMaxSpeedPension(speed=max_speed_observed_by_agent)
            episode_constraint_1[agent] += CalculateSpeedError(speed=avg_speed_observed_by_agent, 
                                                                speed_limit=SPEED_LIMIT,
                                                                lower_speed_limit=SPEED_LIMIT)
            episode_constraint_2[agent] += rewards[agent]   # NOTE That right now, the g2 constraint is the same as the 'queue' model

                
        obses = next_obses

    return episode_rewards, episode_constraint_1, episode_constraint_2
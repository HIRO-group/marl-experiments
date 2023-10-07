"""
sumo_quick_visualize.py
    
Description:
    Script for quickly visualizing a given set of route and net files
    NOTE: this file just executes a random policy on the environment

Usage:
    python sumo_quick_visualize.py
"""

import sumo_rl
from sumo_custom_observation import CustomObservationFunction

# route = '/home/jmiceli/workspace/IndependentStudy/marl-experiments/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml'
# net_file = '/home/jmiceli/workspace/IndependentStudy/marl-experiments/nets/4x4-Lucas/4x4.net.xml'
# route = '/home/jmiceli/workspace/IndependentStudy/marl-experiments/nets/2x2grid/2x2.rou.xml'
# net_file = '/home/jmiceli/workspace/IndependentStudy/marl-experiments/nets/2x2grid/2x2.net.xml'
route = '/home/jmiceli/workspace/IndependentStudy/marl-experiments/nets/3x3grid/routes14000.rou.xml'
net_file = '/home/jmiceli/workspace/IndependentStudy/marl-experiments/nets/3x3grid/3x3Grid2lanes.net.xml'

gui = True
seconds = 1000
min_green = 10
max_green = 50
env = sumo_rl.parallel_env(net_file=net_file, 
                        route_file=route,
                        use_gui=gui,
                        num_seconds=seconds,
                        add_system_info=True,
                        add_per_agent_info=True,
                        reward_fn='queue',
                        observation_class=CustomObservationFunction,
                        sumo_warnings=False)

env.reset()


agents = env.possible_agents
action_spaces = env.action_spaces
observation_spaces = env.observation_spaces
actions = {agent: None for agent in agents}

action_spaces = env.action_spaces
print(" > action_spaces:\n {}".format(action_spaces))

observation_spaces = env.observation_spaces
for agent in agents:
    print(f" > observation_spaces[{agent}] shape:\n {observation_spaces[agent].shape}")

for step in range(seconds):
    for agent in agents:
        actions[agent] = action_spaces[agent].sample()

    next_obses, rewards, dones, truncated, info = env.step(actions)
    for agent in agents:
        print(f" >> next_obses[{agent}: \n {next_obses[agent]}]")

env.close()
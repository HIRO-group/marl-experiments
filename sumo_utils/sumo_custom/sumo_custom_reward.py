"""
Function for defininging the custom sumo reward using experiment arguments
"""

from types import FunctionType
import sys

from sumo_rl import TrafficSignal
from .sumo_custom_reward_avg_speed_limit import AverageSpeedLimitReward
from .sumo_custom_reward_max_speed_limit import MaxSpeedRewardFunction



def CreateSumoReward(args) -> str | FunctionType:
    """
    Function for defining the sumo reward function (either custom or built-in)

    :param args: Config args object
    :returns Either a string definining built in reward function or a custom function 
    """


    # Default value is assumed to be one of the default sumo reward strings
    # https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/traffic_signal.py#L308

    if (args.sumo_reward == "custom"):
        print (f" > Using CUSTOM reward")
        reward_function = MaxSpeedRewardFunction(args.sumo_max_speed_threshold, args.sumo_min_speed_threshold)

    elif (args.sumo_reward == "custom-average-speed-limit"):
        print (f" > Using CUSTOM AVERAGE SPEED LIMIT reward")
        reward_function = AverageSpeedLimitReward(args.sumo_average_speed_limit)

    elif (args.sumo_reward in TrafficSignal.reward_fns.keys()):
        # Use a built in function
        print (f" > Using standard reward: '{args.sumo_reward}'")
        reward_function = args.sumo_reward

    else:
        print(f" > Unrecogrnized reward function provided: {args.sumo_reward}")
        sys.exit(1)


    return reward_function
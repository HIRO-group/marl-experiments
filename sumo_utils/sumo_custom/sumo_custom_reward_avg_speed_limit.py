"""
sumo_custom_reward_avg_speed_limit.py

Description:
    Implementation of custom reward function for the sumo-rl environment. The reward is defined as the
    negative difference between the average observed speed and a speed threshold. This function can be
    provided to the env using the `reward_fn` argument.
"""

from types import FunctionType
from sumo_rl import TrafficSignal

from .calculate_speed_control import CalculateSpeedError

def AverageSpeedLimitReward(speed_limit:float) -> FunctionType:
    """
    Return the negative of difference between the average observered speed of all vehicles at the intersection and a threshold 
    If there are no vehicles in the intersection, returns 0.0
    """
        
    def reward_function(ts:TrafficSignal):
        # Get all vehicles at the intersection
        vehs = ts._get_veh_list()

        if len(vehs) == 0:
            # If there are no vehicles in the intersection, we want to return 0
            return 0.0

        avg_speed = 0.0
        for v in vehs:
            avg_speed += ts.sumo.vehicle.getSpeed(v)

        # Average speed of all vehicles in the intersection
        avg_speed = avg_speed/len(vehs)

        overage = CalculateSpeedError(speed=avg_speed,
                                    speed_limit=speed_limit,
                                    lower_speed_limit=speed_limit)
        return overage
    
    return reward_function
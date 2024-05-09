"""
sumo_custom_reward_avg_speed_limit.py

Description:

    
"""

from sumo_rl import TrafficSignal

from calculate_speed_control import CalculateSpeedError

def AverageSpeedLimitReward(ts:TrafficSignal):
        """

        """
        
        # TODO: Config
        SPEED_LIMIT = 7.0

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
                                    speed_limit=SPEED_LIMIT,
                                    lower_speed_limit=SPEED_LIMIT)

        return overage
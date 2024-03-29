"""
sumo_custom_reward.py

Description:

    
"""

from sumo_rl import TrafficSignal
import numpy as np

from calculate_speed_control import CalculateMaxSpeedOverage

def MaxSpeedRewardFunction(ts:TrafficSignal):
        """
        Return the "pension" (i.e. difference) between the max observered speed of all vehicles at the intersection and a threshold 
        If there are no vehicles in the intersection, returns 0.0
        """
        
        # TODO: make this configurable
        SPEED_THRESHOLD = 13.89 # Chosen based on the "speed" configuration parameter set in many of the defaul .net files
                                # Note this value is actually never observed during training 
        # SPEED_THRESHOLD = 16.0  # Chosen because it is between the minimum max observed speed (15.11) and the max max observed speed during training
                                # Using normal reward function (18.44)
        # SPEED_THRESHOLD = 18.44 # maximum observed speed during training using normal reward function
        
        # SPEED_THRESHOLD = 10.0
    
        # SPEED_THRESHOLD = 20.0

        LOWER_SPEED_THRESHOLD = 1.0
        # LOWER_SPEED_THRESHOLD = 0.01
        # LOWER_SPEED_THRESHOLD = 5.0
        # LOWER_SPEED_THRESHOLD = 0.0

        max_speed = 0.0

        # Get all vehicles at the intersection
        vehs = ts._get_veh_list()
        if len(vehs) == 0:
            max_speed = 0.0

        # Find the max speed of all vehicles in the intersection
        # TODO: We could consider average speed here as well         
        for v in vehs:
            speed = ts.sumo.vehicle.getSpeed(v)
            if speed > max_speed:
                max_speed = speed
        
        overage = CalculateMaxSpeedOverage(max_speed=max_speed,
                                           speed_limit=SPEED_THRESHOLD,
                                           lower_speed_limit=LOWER_SPEED_THRESHOLD)

        return overage

        # if max_speed > SPEED_THRESHOLD:
        #     pension = max_speed - SPEED_THRESHOLD
        #     # If the max speed is greater than then threshold, return the negative 
        #     # of the pension (i.e. difference)
        #     return (-1.0 * np.sqrt(pension))
        
        # # TODO: make lower bound configurable?
        # elif max_speed <= 5.0: 
        #     pension = SPEED_THRESHOLD - max_speed
        #     return (-1.0 * np.sqrt(pension))
        
        # else:
        #     # If the max speed is within the bounds, just return 0
        #     return 0.0
"""
sumo_custom_reward.py

Description:

    
"""

from sumo_rl import TrafficSignal

def MaxSpeedRewardFunction(ts:TrafficSignal):
        """
        Return the "pension" (i.e. difference) between the max observered speed of all vehicles at the intersection and a threshold 
        If there are no vehicles in the intersection, returns 0.0
        """
        
        SPEED_THRESHOLD = 13.89 # Chosen based on the "speed" configuration parameter set in many of the defaul .net files
        
        max_speed = 0.0
        vehs = ts._get_veh_list()
        if len(vehs) == 0:
            max_speed = 0.0
        for v in vehs:
            speed = ts.sumo.vehicle.getSpeed(v)
            if speed > max_speed:
                max_speed = speed
        
        pension = max_speed - SPEED_THRESHOLD
        
        if pension > 0.0:
            # If the max speed is greater than then threshold, return the negative 
            # of the pension (i.e. difference)
            return (-1.0 * pension)
        
        else:
            # If the max speed is less than the threshold, just return 0
            return 0.0
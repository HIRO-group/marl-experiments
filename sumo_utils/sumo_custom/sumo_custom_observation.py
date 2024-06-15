"""
sumo_custom_observation.py

Description:
    Custom obsevation class that adds a "max speed" variable to the end of array returned in the default observation. 
    The max speed value indicates the maximum speed observed from any vehicle by the agent at that step.
    Note that modifying the output of this function changes the size of the neural network produced when training

References:
    Default observation class defined here: https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/observations.py#L28 
    
"""

from sumo_rl import ObservationFunction
from sumo_rl import TrafficSignal
from gymnasium import spaces

import numpy as np

class CustomObservationFunction(ObservationFunction):
    """Custom observation function class to include vehicle speed information in the obs"""
    def __init__(self, ts:TrafficSignal):
        """Initialize the observation function""" 
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the observation.
        (modified from default observation function)"""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        # pressure = [self.ts.get_pressure()] # MODIFYING THE OUTPUT OF THIS FUNCTION REQUIRES RETRAINING
        max_speed = [self.get_max_speed()]
        avg_speed = [self.get_true_average_speed()]   # Get the average speed reward for this agent at this step
        # observation = np.array(phase_id + min_green + density + queue + pressure + max_speed, dtype=np.float32)
        observation = np.array(phase_id + min_green + density + queue + avg_speed + max_speed, dtype=np.float32)
        return observation
    
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes) + 1 + 1, dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes) + 1 + 1, dtype=np.float32),
        )


    def get_max_speed(self):
        """Get the max speed of all vehicles at the intersection. If there are no vehicles in the
        intersection, it returns 0.0"""
        max_speed = 0.0
        vehs = self.ts._get_veh_list()
        if len(vehs) == 0:
            return 0.0
        for v in vehs:
            speed = self.ts.sumo.vehicle.getSpeed(v)
            if speed > max_speed:
                max_speed = speed
        return max_speed
    
    def get_true_average_speed(self):

        vehs = self.ts._get_veh_list()

        if len(vehs) == 0:
            # If there are no vehicles in the intersection, we want to return 0
            return 0.0

        avg_speed = 0.0
        for v in vehs:
            avg_speed += self.ts.sumo.vehicle.getSpeed(v)

        # Average speed of all vehicles in the intersection
        avg_speed = avg_speed/len(vehs)

        return avg_speed
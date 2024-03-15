import numpy as np



def CalculateMaxSpeedOverage(max_speed:float, 
                             speed_limit:float,
                             lower_speed_limit:float=1.0) -> float:
    """
    # TODO Update description
    Calculate how much the agents' max speeds exceeded some speed limit
    :param max_speed: Max speed of all cars observed by the agent (assumed at a single step)
    :param speed_limit: User defined threshold over which the overage is calculated
    :returns -1 times how much the max speed exceeded the speed limit
    """

    if (max_speed > speed_limit):
        pension = max_speed - speed_limit
        # If the max speed is greater than then threshold, return the negative 
        # of the pension (i.e. difference)
        return (-1.0 * np.sqrt(pension))
        
    # DEBUGGING:
    ##### elif max_speed <= 5.0:  
    ##### Also just take out this elif statement, do we still have "flipped" plot without the lower bound???

    elif max_speed <= lower_speed_limit:   
        pension = speed_limit - max_speed
        return (-1.0 * np.sqrt(pension))
    
    else:
        # If the max speed is within the bounds, just return 0
        return 0.0
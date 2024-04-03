

def CalculateSpeedError(speed:float, 
                             speed_limit:float,
                             lower_speed_limit:float=1.0) -> float:
    """
    Calculate how much the agent's observed speeds exceeded some speed limit
    :param speed: Max speed of all cars observed by the agent (assumed at a single step)
    :param speed_limit: User defined threshold over which the error is calculated
    :param lower_speed_limit: User defined threshold below which error is calculated
    :returns -1 times how much the speed exceededs the interval of [lower_speed_limit, speed_limit)
    """

    if (speed > speed_limit):
        pension = speed - speed_limit      


    elif (speed <= lower_speed_limit):  
        pension = lower_speed_limit - speed

    else:
        # If the max speed is within the bounds, just return 0
        pension = 0.0
        
    return (-1.0 * pension)
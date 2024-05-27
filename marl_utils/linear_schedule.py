

def LinearSchedule(start_e: float, end_e: float, duration: int, t: int):
    """
    Defines a schedule for decaying epsilon during the training procedure

    :param start_e: Starting value of epsilon
    :param end_e: Smallest allowable value that epsilon can have
    :parm duraiton: Total number of time steps
    :param t: Current time step
    :returns: Linearly decayed value of epsilon
    """
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
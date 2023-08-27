

def LinearSchedule(start_e: float, end_e: float, duration: int, t: int):
    """
    Defines a schedule for decaying epsilon during the training procedure
    """
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
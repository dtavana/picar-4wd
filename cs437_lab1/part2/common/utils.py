# Filter for invalid distances returned by the ultrasonic sensor
def is_invalid_distance(distance: float) -> bool:
    '''
    Returns whether the given distance fromthe ultrasonic sensor should be considered invalid
    '''
    return distance < 0
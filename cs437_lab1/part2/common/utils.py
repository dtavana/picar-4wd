# Maximum valid range that the ultrasonic sensor can read (taken from https://cdn.sparkfun.com/datasheets/Sensors/Proximity/HCSR04.pdf)
MAX_VALID_DISTANCE = 400

# Filter for invalid distances returned by the ultrasonic sensor

def is_invalid_distance(distance):
    '''
    Returns whether the given distance fromthe ultrasonic sensor should be considered invalid
    '''
    return not (0 < distance <= MAX_VALID_DISTANCE)
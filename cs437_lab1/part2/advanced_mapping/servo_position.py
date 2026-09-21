import picar_4wd as fc
import time

from ..common import utils

# Time to let the servo settle between setting angle and measuring distance
SERVO_SETTLE_TIME = 0.1
# Time to let the ultrasonic sensor settle between distance measurements
ULTRASONIC_SETTLE_TIME = 0.05

class ServoPosition:
    def __init__(self, angle, name=''):
        self.angle = angle
        self.name = name


    def set_angle(self):
        '''
        Sets the servo to the target angle
        '''
        fc.servo.set_angle(self.angle)
    
    def get_distance(self):
        '''
        Retrieves the distance reported by the ultrasonic sensor
        '''
        return fc.us.get_distance()

    def get_distance_at_angle(self, sample_count=3):
        '''
        Returns the measured distance at the target angle returning the specified number of samples

        If `invalid_distance` is True then the cell should be marked as state unknown (not clear or obstacle detected)
        '''
        self.set_angle()
        time.sleep(SERVO_SETTLE_TIME)
        readings = []
        for _ in range(sample_count):
            distance = self.get_distance()
            time.sleep(ULTRASONIC_SETTLE_TIME)
            readings.append({
                "distance": distance,
                "invalid_distance": utils.is_invalid_distance(distance),
            })
        return readings
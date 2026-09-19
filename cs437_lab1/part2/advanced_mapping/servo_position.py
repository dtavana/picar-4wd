import picar_4wd as fc

from ..common import utils


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
        Returns the measured distance at the target angle, if it should be considered an invalid distance, and whether it should consider an obstacle detected

        If `invalid_distance` is True then the cell should be marked as state unknown (not clear or obstacle detected)
        '''
        distance = fc.get_distance_at(self.angle)
        return {
            "distance": distance,
            "invalid_distance": utils.is_invalid_distance(distance),
        }
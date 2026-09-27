HEADING_NAMES = ["north", "east", "south", "west"]
# Unit vectors (x, y) for each heading, where +x is east (right of the starting
# direction) and +y is north (the direction the car faces when it starts)
HEADING_VECTORS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


class Pose:
    '''
    Dead reckoned position of the car in world coordinates (cm), relative to where it started.
    The car starts at (0, 0) facing north. Headings are restricted to 90 degree increments
    (0 = north, 1 = east, 2 = south, 3 = west) because the car only makes 90 degree turns.
    '''
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.heading = 0

    def turn(self, quarter_turns):
        '''
        Records a turn of `quarter_turns` 90 degree turns to the right (negative for left)
        '''
        self.heading = (self.heading + quarter_turns) % 4

    def move(self, distance):
        '''
        Records moving `distance` cm in the current heading (negative for reversing)
        '''
        x_unit, y_unit = HEADING_VECTORS[self.heading]
        self.x += x_unit * distance
        self.y += y_unit * distance

    def world_to_local(self, x, y):
        '''
        Converts a world coordinate to (forward, right) cm relative to the car's current pose
        '''
        x_delta, y_delta = x - self.x, y - self.y
        forward_x, forward_y = HEADING_VECTORS[self.heading]
        right_x, right_y = HEADING_VECTORS[(self.heading + 1) % 4]
        forward = x_delta * forward_x + y_delta * forward_y
        right = x_delta * right_x + y_delta * right_y
        return forward, right

    def __str__(self):
        return f"({self.x:.1f}, {self.y:.1f}) facing {HEADING_NAMES[self.heading]}"

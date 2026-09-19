import picar_4wd as fc
from numpy import median

from .map_array import MapArray, GridState
from .servo_position import ServoPosition

# Number of times to take a distance sample at a given angle
DISTANCE_SAMPLE_COUNT = 3
# Quorom of valid distance samples needed to update the map array
DISTANCE_QUORUM_COUNT = 2



class Scanner:
    def __init__(self, left_scan_max, right_scan_max, scan_increment, servo_offset):
        self.left_scan_max = left_scan_max
        self.right_scan_max = right_scan_max
        if self.right_scan_max <= self.left_scan_max:
            raise ValueError(f"right_scan_max ({self.right_scan_max} degrees) must be greater than left_scan_max ({self.left_scan_max} degrees)")
        self.scan_increment = scan_increment
        self.servo_offset = servo_offset

        self.map_array = MapArray()
        self.servo_positions = [ServoPosition(x) for x in range(self.left_scan_max, self.right_scan_max + 1, self.scan_increment)]
        self.init_servo()

    def init_servo(self):
        '''
        Initializes the servo before use
        '''
        fc.servo.offset = self.servo_offset
        fc.servo.set_angle(0)

    def perform_scan(self):
        '''
        Performs a scan for all initialized servo positions
        '''
        scan_results = []
        for pos in self.servo_positions:
            # Collect several samples at given position before updating array
            readings = []
            for _ in range(DISTANCE_SAMPLE_COUNT):
                readings.append(pos.get_distance())
            filtered_readings = [x for x in readings if not x["invalid_distance"]]
            if len(filtered_readings) >= DISTANCE_QUORUM_COUNT:
                # We have enough valid distance sample readings, update map array from a median distance
                distance = median([x["distance"] for x in filtered_readings])
            else:
                distance = GridState.UNKNOWN
            scan_results.append({"position": pos, "distance": distance})
        return scan_results


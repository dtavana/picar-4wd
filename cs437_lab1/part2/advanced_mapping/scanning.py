from numpy import median

import picar_4wd as fc

from .map_array import GridState
from .servo_position import ServoPosition

# Degree for left most scan
LEFT_SCAN_MAX = -60
# Degree for right most scan
RIGHT_SCAN_MAX = 60
# Number of degrees to increment in between scans
SCAN_INCREMENT = 5

# Number of times to take a distance sample at a given angle
DISTANCE_SAMPLE_COUNT = 3
# Quorom of valid distance samples needed to update the map array
DISTANCE_QUORUM_COUNT = 2
# Maxmimum allowed distance delta in cm between valid scans at a given angle
MAX_ANGLE_DISANCE_DELTA = 5

class Scanner:
    def __init__(self, servo_offset):
        self.servo_offset = servo_offset
        self.servo_positions = [ServoPosition(x) for x in range(LEFT_SCAN_MAX, RIGHT_SCAN_MAX + 1, SCAN_INCREMENT)]
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
            readings = pos.get_distance_at_angle(DISTANCE_SAMPLE_COUNT)
            filtered_readings = [x for x in readings if not x["invalid_distance"]]
            distance = GridState.UNKNOWN
            if len(filtered_readings) >= DISTANCE_QUORUM_COUNT:
                # We have enough valid distance sample readings
                distances = [x["distance"] for x in filtered_readings]
                if max(distances) - min(distances) <= MAX_ANGLE_DISANCE_DELTA:
                    distance = median(distances)   
            print(
                f"{pos.angle:+3}°: "
                f"raw={[x['distance'] for x in readings]}, "
                f"valid={len(filtered_readings)}, "
                f"selected={distance}"
            )     
            scan_results.append({"position": pos, "distance": distance})
        fc.servo.set_angle(0)
        return scan_results


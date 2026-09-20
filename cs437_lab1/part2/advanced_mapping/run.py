import sys
import time

import picar_4wd as fc

from .map_array import MapArray
from .scanning import Scanner


def main(servo_offset):
    print("CS437 Lab 1 Part 2 Advanced Mapping Test")
    print("Use Ctrl+C to stop at any time")
    print("------------------------------")

    
    scanner = Scanner(servo_offset)
    map_array = MapArray()

    while True:
        scan = scanner.perform_scan()
        for result in scan:
            print(
                f'{result["position"].angle:+3}°: '
                f'{result["distance"]:.1f} cm'
            )
        map_array.update_from_scan(scan)
        map_array.visualize()
        print(f"\n{'-' * map_array.grid.shape[0]}\n")
        time.sleep(10)

if __name__ == "__main__":
    # Parse servo_offset command line
    servo_offset = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    try:
        main(servo_offset)
    except KeyboardInterrupt:
        print("\nStopping")
    finally:
        fc.servo.set_angle(0)
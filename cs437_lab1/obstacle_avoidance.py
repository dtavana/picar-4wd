import picar_4wd as fc
import time
import random
import sys

# Obstacles within 30cm should be avoided
OBSTACLE_CLOSE_THRESHOLD = 30 
# Time to sleep between scanning angles
SWEEP_SLEEP = 0.25 
# Time to sleep between different movements
MOVEMENT_SLEEP = 0.1 

# Speed when moving forward and not avoiding obstacles
FORWARD_SPEED = 30 
# Speed when turning to avoid obstacle
TURN_SPEED = 20 
# Speed when backing up when obstacle is avoided
REVERSE_SPEED = 20 

# How long to reverse car for when obstacle detected
REVERSE_TIME = 0.5 
# Amount of time to turn the vehicle
TURN_TIME = 1 

# How many historical center distance readings to keep
CENTER_DISTANCE_SAMPLE_COUNT = 8
# Threshold for how much the difference between max and min 
# historical distance readings means the car is stuck
CENTER_DISTANCE_STUCK_THRESHOLD = 4 
# List to hold historical center readings
CENTER_DISTANCE_HISTORY = [] 

class ServoPosition:
    def __init__(self, name, angle, turn_fn):
        self.name = name
        self.angle = angle
        self.turn_fn = turn_fn
        self.distance = 0

    def turn(self):
        if self.turn_fn is None:
            return
        self.turn_fn(TURN_SPEED)
        time.sleep(TURN_TIME)
        fc.stop()
    
    def set_angle(self):
        fc.servo.set_angle(self.angle)
    
    def get_distance(self):
        self.distance = fc.get_distance_at(self.angle)
        return self.distance
    
    def is_valid_direction(self):
        distance = self.get_distance()
        # A distance of -2 means the ultrasonic sensor timed out while measuring (see ultrasonic.py lines 35-38).
        # fc.get_status_at() treats -2 as a "far" distance, therefore the same logic is applied here.
        is_valid = distance == -2 or distance > OBSTACLE_CLOSE_THRESHOLD
        return is_valid

FRONT_LEFT_SERVO_POSITION = ServoPosition("front-left", -25, None)
FAR_LEFT_SERVO_POSITION = ServoPosition("left", -45, fc.turn_left)
CENTER_SERVO_POSITION = ServoPosition("center", 0, None)
FRONT_RIGHT_SERVO_POSITION = ServoPosition("front-right", 25, None)
FAR_RIGHT_SERVO_POSITION = ServoPosition("right", 45, fc.turn_right)

MOVING_SCAN_POSITIONS = [
    FRONT_LEFT_SERVO_POSITION,
    CENTER_SERVO_POSITION,
    FRONT_RIGHT_SERVO_POSITION,
    CENTER_SERVO_POSITION
]

OBSTACLE_SCAN_POSITIONS = [
    FAR_LEFT_SERVO_POSITION,
    FAR_RIGHT_SERVO_POSITION
]

def scan_for_valid_positions():
    valid_positions = []
    for pos in OBSTACLE_SCAN_POSITIONS:
        is_valid = pos.is_valid_direction()
        print(f"{pos.name}: {pos.distance:.1f} cm")
        if is_valid:
            valid_positions.append(pos)
        time.sleep(SWEEP_SLEEP)
    return valid_positions

def reverse():
    fc.backward(REVERSE_SPEED)
    time.sleep(REVERSE_TIME)
    fc.stop()

def avoid_obstacle():
    # Do an initial scan to see if a position is available to move in now
    valid_positions = scan_for_valid_positions()
    if not valid_positions:
        # From current position no initial direction open, reverse and retry
        print("NO VALID DIRECTIONS FOUND DURING INITIAL SWEEP, BACKING UP AND RESCANNING")
        reverse()
        # Rescan from reversed position
        valid_positions = scan_for_valid_positions()
    if valid_positions:
        print(f"VALID DIRECTIONS FOUND DURING SWEEP: {[x.name for x in valid_positions]}")
        target_pos = random.choice(valid_positions)
    else:
        # After rescan still no valid positions, pick one at random
        print("NO VALID DIRECTION FOUND DURING SWEEP, PICKING RANDOM DIRECTION")
        target_pos = random.choice([FAR_LEFT_SERVO_POSITION, FAR_RIGHT_SERVO_POSITION])

    print(f"MOVING IN {target_pos.name} DIRECTION")
    # Turn in the direction of the chosen position
    target_pos.turn()
    # Reset servo to center
    CENTER_SERVO_POSITION.set_angle()
    time.sleep(MOVEMENT_SLEEP)

def process_historical_center_readings():
    if CENTER_SERVO_POSITION.distance <= 0:
        CENTER_DISTANCE_HISTORY.clear()
        return False
    # Normal distance reading, use historical data
    CENTER_DISTANCE_HISTORY.append(CENTER_SERVO_POSITION.distance)
    if len(CENTER_DISTANCE_HISTORY) > CENTER_DISTANCE_SAMPLE_COUNT:
        # Pop oldest distance reading
        CENTER_DISTANCE_HISTORY.pop(0)
    if len(CENTER_DISTANCE_HISTORY) == CENTER_DISTANCE_SAMPLE_COUNT:
        # Once we have enough readings, check if we are actually stuck
        delta_distance_change = max(CENTER_DISTANCE_HISTORY) - min(CENTER_DISTANCE_HISTORY)
        if delta_distance_change <= CENTER_DISTANCE_STUCK_THRESHOLD:
            print("DETECTED CAR IS STUCK, BACKING UP AND MOVING IN NEW DIRECTION")
            fc.stop()
            CENTER_DISTANCE_HISTORY.clear()
            # First reverse as obstacles weren't detected at current position, then scan for obstacles and make a turn
            reverse()
            avoid_obstacle()
            return True
    return False

def main(servo_offset=0):
    print("CS437 Lab 1 Part 1 Obstacle Avoidance")
    print("Use Ctrl+C to stop at any time")
    print("------------------------------")

    # Set the servo offset
    fc.servo.offset = servo_offset

    # Reset servo to center
    CENTER_SERVO_POSITION.set_angle()

    while True:
        obstacle = False
        stuck = False
        fc.forward(FORWARD_SPEED)
        # Scan all "forward positions" while moving
        for pos in MOVING_SCAN_POSITIONS:
            is_valid = pos.is_valid_direction()
            if not is_valid:
                # If any "forward positions" have an obstacle we should note it and ultimately stop
                print(f"OBSTACLE DETECTED AT {pos.name}: {pos.distance:.1f} cm")
                obstacle = True
                break
            if pos is CENTER_SERVO_POSITION:
                # If we are currently scanning the center position, process the reading for stuck detection
                stuck = process_historical_center_readings()
                if stuck:
                    break
        if not obstacle or stuck:
            continue
        fc.stop()
        # Turning the car, historical center data no longer relevant
        CENTER_DISTANCE_HISTORY.clear()
        avoid_obstacle()
            
if __name__ == "__main__":
    # Parse servo_offset command line
    servo_offset = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    try:
        main(servo_offset)
    except KeyboardInterrupt:
        print("\nStopping")
    finally:
        fc.stop()
        CENTER_SERVO_POSITION.set_angle()
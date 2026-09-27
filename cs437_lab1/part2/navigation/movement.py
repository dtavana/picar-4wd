import time

import picar_4wd as fc

from ..common import utils

# Power used when driving forward along a path
FORWARD_SPEED = 30
# Power used when turning in place
TURN_SPEED = 30
# Power used when reversing away from obstacles
REVERSE_SPEED = 20

# Calibration values for time based movement, tune these with calibrate.py on the
# surface you are driving on (they change with battery level and floor type)
# How many cm the car travels per second at FORWARD_SPEED / REVERSE_SPEED
FORWARD_CM_PER_SECOND = 35.0
REVERSE_CM_PER_SECOND = 28.0
# How long to turn in place at TURN_SPEED to rotate 90 degrees right / left
# (tuned separately because the motors may not be equally strong in both directions)
TURN_90_TIME = 1.5
TURN_LEFT_90_TIME = 2.0
# Power difference between the left and right wheels to make the car drive straight.
# Positive speeds up the left wheels and slows the right (fixes drifting left),
# negative does the opposite (fixes drifting right)
DRIFT_CORRECTION = 0

# Time to let the car settle after each movement
MOVEMENT_SLEEP = 0.2
# While driving forward, stop early if the ultrasonic sensor reads an obstacle this close in cm
EMERGENCY_STOP_DISTANCE = 10
# Number of close readings in a row needed for an emergency stop, so a single noisy reading
# from the ultrasonic sensor doesn't stop the car for nothing
EMERGENCY_STOP_READINGS = 3
# Time between ultrasonic readings while driving forward
DISTANCE_POLL_INTERVAL = 0.05


def drive_straight(power):
    '''
    Drives all wheels at `power` (negative for reverse), applying DRIFT_CORRECTION between the sides
    '''
    direction = 1 if power >= 0 else -1
    left_power = power + direction * DRIFT_CORRECTION
    right_power = power - direction * DRIFT_CORRECTION
    fc.left_front.set_power(left_power)
    fc.left_rear.set_power(left_power)
    fc.right_front.set_power(right_power)
    fc.right_rear.set_power(right_power)


def forward(distance, should_stop=None):
    '''
    Drives forward `distance` cm, stopping early if an obstacle gets within EMERGENCY_STOP_DISTANCE
    or the optional `should_stop` function returns True (e.g. a person was detected).
    The ultrasonic servo is expected to be facing forward.

    Returns the estimated distance in cm actually travelled.
    '''
    duration = distance / FORWARD_CM_PER_SECOND
    start_time = time.time()
    close_readings = 0
    drive_straight(FORWARD_SPEED)
    try:
        while time.time() - start_time < duration:
            obstacle_distance = fc.us.get_distance()
            if not utils.is_invalid_distance(obstacle_distance) and obstacle_distance < EMERGENCY_STOP_DISTANCE:
                close_readings += 1
            else:
                close_readings = 0
            if close_readings >= EMERGENCY_STOP_READINGS:
                print(f"EMERGENCY STOP: obstacle {obstacle_distance:.1f} cm ahead")
                break
            if should_stop is not None and should_stop():
                print("STOPPING: should_stop requested")
                break
            time.sleep(DISTANCE_POLL_INTERVAL)
    finally:
        fc.stop()
    elapsed = min(time.time() - start_time, duration)
    time.sleep(MOVEMENT_SLEEP)
    return elapsed * FORWARD_CM_PER_SECOND


def reverse(distance):
    '''
    Reverses `distance` cm
    '''
    drive_straight(-REVERSE_SPEED)
    time.sleep(distance / REVERSE_CM_PER_SECOND)
    fc.stop()
    time.sleep(MOVEMENT_SLEEP)


def turn(quarter_turns):
    '''
    Turns in place by `quarter_turns` 90 degree turns to the right (negative for left).
    A turn of 3 quarter turns is done as a single left turn.
    '''
    quarter_turns %= 4
    if quarter_turns == 3:
        turn_fn, count, turn_time = fc.turn_left, 1, TURN_LEFT_90_TIME
    else:
        turn_fn, count, turn_time = fc.turn_right, quarter_turns, TURN_90_TIME
    for _ in range(count):
        turn_fn(TURN_SPEED)
        time.sleep(turn_time)
        fc.stop()
        time.sleep(MOVEMENT_SLEEP)

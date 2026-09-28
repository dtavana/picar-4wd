import argparse
import math
import sys
import time

import numpy as np
import picar_4wd as fc

from ..advanced_mapping.map_array import CELL_SIZE, GridState, MapArray
from ..advanced_mapping.scanning import Scanner
from . import movement
from .astar import FORWARD, find_path, inflate_obstacles, nearest_free_cell, path_to_segments
from .pose import Pose

# Radius in cm to pad around every obstacle so the car's body fits through gaps.
CLEARANCE_RADIUS = 15
# The goal counts as reached when the car is within this many cm of it in both axes
GOAL_TOLERANCE = 10
# Distance in cm to follow a planned path before rescanning and replanning
RESCAN_DISTANCE = 40
# Path segments shorter than this in cm are not worth turning for
MIN_SEGMENT_DISTANCE = 3
# Distance in cm to back up when no path can be found
BACKUP_DISTANCE = 10
# Number of consecutive failed plans before turning towards the goal to look elsewhere
MAX_FAILED_PLANS = 2
# Maximum number of scan/plan/move cycles before giving up
MAX_ITERATIONS = 50
# Time between checks while paused (e.g. waiting for a person to move)
PAUSE_POLL_INTERVAL = 0.5

# Object detection labels that make the car wait in place until they are gone
PAUSE_LABELS = ["person"]
# Minimum detection score for a label to count as seen (higher means fewer false pauses)
DETECTION_MIN_SCORE = 0.6
# A label counts as still in view for this many seconds after it was last detected,
# so a single missed frame doesn't make the car start moving again
DETECTION_HOLD_TIME = 1.5

# Traffic sign the car obeys (Step 9): come to a full stop, wait, then continue the route
STOP_SIGN_LABEL = "stop sign"
# Stop signs further away score lower than people up close, so use a lower threshold
STOP_SIGN_MIN_SCORE = 0.5
# How long to stay stopped at a stop sign in seconds
STOP_SIGN_WAIT_TIME = 3
# Only stop once the sign is close: its box must be at least this fraction of the camera
# image height (about 0.25 is 60-70 cm away for a 15 cm sign). Tune using the CAMERA SEES line
STOP_SIGN_MIN_HEIGHT = 0.60
# A stop sign only makes the car stop once. It counts as a new sign again after it has
# been out of view for this many seconds
STOP_SIGN_REARM_TIME = 3

# Maximum number of detections listed on each iteration's CAMERA SEES line
MAX_DETECTIONS_SHOWN = 5
# Each character of the printed map shows a block of this many x this many cells
MAP_BLOCK_SIZE = 3


def goal_to_cell(map_array, forward, right):
    '''
    Converts a goal relative to the car into grid coordinates, clamping goals beyond the edge
    of the map to the nearest cell on the edge so we can still route towards them
    '''
    row = map_array.car_origin_row - int(round(forward / CELL_SIZE))
    column = map_array.car_origin_column + int(round(right / CELL_SIZE))
    row = min(max(row, 0), map_array.grid.shape[0] - 1)
    column = min(max(column, 0), map_array.grid.shape[1] - 1)
    return row, column


def turn_towards(pose, right):
    '''
    Turns the car towards the side the goal is on (turning around if it is directly behind)
    '''
    if right > 0:
        quarter_turns = 1
    elif right < 0:
        quarter_turns = -1
    else:
        quarter_turns = 2
    movement.turn(quarter_turns)
    pose.turn(quarter_turns)


def close_stop_sign(detector):
    '''
    Returns the most confident stop sign in the latest camera frame that is both confident
    enough (STOP_SIGN_MIN_SCORE) and close enough (STOP_SIGN_MIN_HEIGHT), or None
    '''
    signs = [d for d in detector.latest()
             if d['label'] == STOP_SIGN_LABEL
             and d['score'] >= STOP_SIGN_MIN_SCORE
             and d['height'] >= STOP_SIGN_MIN_HEIGHT]
    return max(signs, key=lambda d: d['score'], default=None)


def make_detection_pause(detector):
    '''
    Returns a should_pause function implementing the car's traffic rules:
      - when close to a stop sign, come to a full stop for STOP_SIGN_WAIT_TIME, then continue
      - wait in place while any PAUSE_LABELS (people) are in view of the camera
    '''
    was_paused = False
    # Time the current stop sign stop ends, or None when not stopped for a sign
    stop_until = None
    # True once we've stopped for the stop sign currently in view
    stopped_for_sign = False

    def should_pause():
        nonlocal was_paused, stop_until, stopped_for_sign
        if not detector.is_running():
            # Don't keep driving blind if the camera or model crashed
            raise RuntimeError(f"Object detection stopped: {detector.error!r}")

        now = time.time()
        if stop_until is not None:
            if now < stop_until:
                return True
            print("STOP SIGN: done waiting, continuing")
            stop_until = None
        if not detector.seen_recently(STOP_SIGN_LABEL, STOP_SIGN_REARM_TIME):
            # The sign we stopped for is out of view, so the next stop sign counts again
            stopped_for_sign = False
        elif not stopped_for_sign:
            sign = close_stop_sign(detector)
            if sign is not None:
                print(f"STOP SIGN DETECTED ({sign['score']:.2f}, {sign['height']:.0%} of frame height): "
                      f"stopping for {STOP_SIGN_WAIT_TIME} s")
                stopped_for_sign = True
                stop_until = now + STOP_SIGN_WAIT_TIME
                return True

        paused = any(detector.seen_recently(label, DETECTION_HOLD_TIME) for label in PAUSE_LABELS)
        if paused and not was_paused:
            # Log what triggered the pause, useful for spotting false detections
            seen = [f"{d['label']} {d['score']:.2f}" for d in detector.latest() if d['label'] in PAUSE_LABELS]
            print(f"DETECTED: {', '.join(seen) or 'recently seen ' + '/'.join(PAUSE_LABELS)}")
        was_paused = paused
        return paused
    return should_pause


def describe_detections(detector):
    '''
    Returns a one line summary of what the camera currently sees, most confident first.
    Lists everything the model reports, including detections below the pause thresholds.
    '''
    detections = sorted(detector.latest(), key=lambda d: d['score'], reverse=True)
    seen = ", ".join(
        f"{d['label']} {d['score']:.2f}" + (f" (h {d['height']:.0%})" if d['label'] == STOP_SIGN_LABEL else "")
        for d in detections[:MAX_DETECTIONS_SHOWN])
    return f"CAMERA SEES: {seen or 'nothing'} (detection FPS {detector.fps:.1f})"


def wait_while_paused(should_pause):
    if should_pause is None or not should_pause():
        return
    print("PAUSED: waiting before moving")
    while should_pause():
        time.sleep(PAUSE_POLL_INTERVAL)
    print("RESUMING")


def follow_path(segments, pose, should_pause):
    '''
    Drives along the planned (direction, cell_count) segments for up to RESCAN_DISTANCE cm.
    The car always faces local FORWARD at the start because the map is built from its perspective.
    '''
    local_direction = FORWARD
    travelled = 0
    for direction, cell_count in segments:
        remaining = RESCAN_DISTANCE - travelled
        if remaining < MIN_SEGMENT_DISTANCE:
            break
        distance = min(cell_count * CELL_SIZE, remaining)
        quarter_turns = (direction - local_direction) % 4
        if quarter_turns and distance < MIN_SEGMENT_DISTANCE and travelled > 0:
            # Not worth turning for a tiny jog, skip it and keep following the path.
            # The first segment is always driven so the car can't get stuck in place.
            continue
        wait_while_paused(should_pause)
        if quarter_turns:
            movement.turn(quarter_turns)
            pose.turn(quarter_turns)
            local_direction = direction
        print(f"DRIVING {distance:.0f} cm")
        moved = movement.forward(distance, should_pause)
        pose.move(moved)
        travelled += moved
        if moved < distance:
            # Stopped early for an obstacle or a person, wait/replan from a fresh scan
            break


def visualize(map_array, blocked, path, goal_cell):
    '''
    Prints the map with the planned path, goal and obstacle clearance on top.
    Like MapArray.visualize, each character is a MAP_BLOCK_SIZE x MAP_BLOCK_SIZE block of cells
    and completely empty rows at the top are skipped, so the map fits in the terminal.
    A block shows the most important thing in it: car, goal, path, obstacle, clearance, clear, unknown.
    '''
    size = MAP_BLOCK_SIZE
    grid = map_array.grid
    car_row, car_column = map_array.car_origin_row, map_array.car_origin_column
    path_mask = np.zeros(grid.shape, dtype=bool)
    for row, column in path or []:
        path_mask[row, column] = True
    goal_row, goal_column = goal_cell

    print(f"UNKNOWN is ? | CLEAR is . | OCCUPIED is X | CLEARANCE is + | PATH is * | GOAL is G | CAR is C"
          f" | each character is {size * CELL_SIZE}x{size * CELL_SIZE} cm")

    # Skip blocks of rows at the top with nothing to show, stopping before the car
    start_row = 0
    while start_row + size <= car_row:
        rows = slice(start_row, start_row + size)
        has_goal = start_row <= goal_row < start_row + size
        if has_goal or np.any(grid[rows] != GridState.UNKNOWN) or np.any(path_mask[rows]) or np.any(blocked[rows]):
            break
        start_row += size
    # Keep one empty display row as a margin above the map
    start_row = max(0, start_row - size)

    for row in range(start_row, grid.shape[0], size):
        row_characters = []
        for column in range(0, grid.shape[1], size):
            rows, columns = slice(row, row + size), slice(column, column + size)
            block = grid[rows, columns]
            if row <= car_row < row + size and column <= car_column < column + size:
                row_characters.append("C")
            elif row <= goal_row < row + size and column <= goal_column < column + size:
                row_characters.append("G")
            elif np.any(path_mask[rows, columns]):
                row_characters.append("*")
            elif np.any(block == GridState.OCCUPIED):
                row_characters.append("X")
            elif np.any(blocked[rows, columns]):
                row_characters.append("+")
            elif np.any(block == GridState.CLEAR):
                row_characters.append(".")
            else:
                row_characters.append("?")
        print("".join(row_characters))


def navigate(goal_forward, goal_right, servo_offset, should_pause=None, detector=None):
    '''
    Drives to the goal at (goal_forward, goal_right) cm relative to the car's starting pose by
    repeatedly scanning, planning a route with A*, and following part of it.

    `should_pause` is an optional function that returns True while the car should wait in place
    (e.g. while object detection sees a person). `detector` is the optional BackgroundDetector,
    used to print what the camera sees on each iteration.
    '''
    scanner = Scanner(servo_offset)
    map_array = MapArray()
    pose = Pose()
    # World coordinates: +x is right of the starting direction, +y is the starting direction
    goal_x, goal_y = goal_right, goal_forward
    clearance_cells = int(math.ceil(CLEARANCE_RADIUS / CELL_SIZE))
    failed_plans = 0

    for iteration in range(1, MAX_ITERATIONS + 1):
        forward, right = pose.world_to_local(goal_x, goal_y)
        print(f"\n===== ITERATION {iteration}: car at {pose}, goal is {forward:.0f} cm forward, {right:.0f} cm right =====")
        if detector is not None:
            print(describe_detections(detector))
        if abs(forward) <= GOAL_TOLERANCE and abs(right) <= GOAL_TOLERANCE:
            print("GOAL REACHED")
            return True
        if forward < 0:
            # The scanner can only see in front of the car, face the goal before scanning
            print("GOAL IS BEHIND, TURNING TOWARDS IT")
            turn_towards(pose, right)
            continue

        wait_while_paused(should_pause)
        scan = scanner.perform_scan()
        map_array.update_from_scan(scan)

        blocked = inflate_obstacles(map_array.grid == GridState.OCCUPIED, clearance_cells)
        start_cell = (map_array.car_origin_row, map_array.car_origin_column)
        goal_cell = goal_to_cell(map_array, forward, right)
        target_cell = nearest_free_cell(blocked, goal_cell)
        path = None
        if target_cell is not None:
            path = find_path(blocked, start_cell, target_cell)
        visualize(map_array, blocked, path, goal_cell)

        if not path or len(path) < 2:
            failed_plans += 1
            print(f"NO PATH FOUND ({failed_plans} in a row), BACKING UP")
            movement.reverse(BACKUP_DISTANCE)
            pose.move(-BACKUP_DISTANCE)
            if failed_plans >= MAX_FAILED_PLANS:
                print("TURNING TOWARDS GOAL TO LOOK FOR ANOTHER ROUTE")
                turn_towards(pose, right if right != 0 else 1)
                failed_plans = 0
            continue

        failed_plans = 0
        segments = path_to_segments(path)
        print(f"PATH: {len(path) - 1} cells, segments (direction, cells): {segments}")
        follow_path(segments, pose, should_pause)

    print("GAVE UP: too many iterations without reaching the goal")
    return False


def main():
    parser = argparse.ArgumentParser(description="CS437 Lab 1 Part 2 Self-Driving Navigation (A*)")
    parser.add_argument("forward", type=float, help="Goal distance forward from the start in cm")
    parser.add_argument("right", type=float, help="Goal distance right of the start in cm (negative for left)")
    parser.add_argument("servo_offset", type=int, nargs="?", default=0, help="Ultrasonic servo offset")
    parser.add_argument("--no-detection", action="store_true", help="Drive without the camera (never pauses for people)")
    args = parser.parse_args()
    # Print each line immediately, otherwise output piped through tee shows up in delayed chunks
    sys.stdout.reconfigure(line_buffering=True)

    print("CS437 Lab 1 Part 2 Self-Driving Navigation")
    print("Use Ctrl+C to stop at any time")
    print("------------------------------")
    detector = None
    should_pause = None
    try:
        if not args.no_detection:
            # Imported here so navigation can run without TensorFlow/camera when --no-detection is used
            from .background_detector import BackgroundDetector
            print("Starting object detection...")
            detector = BackgroundDetector(
                min_score=DETECTION_MIN_SCORE, label_min_scores={STOP_SIGN_LABEL: STOP_SIGN_MIN_SCORE}).start()
            should_pause = make_detection_pause(detector)
        navigate(args.forward, args.right, args.servo_offset, should_pause, detector)
    except KeyboardInterrupt:
        print("\nStopping")
    finally:
        fc.stop()
        fc.servo.set_angle(0)
        if detector is not None:
            detector.stop()


if __name__ == "__main__":
    main()

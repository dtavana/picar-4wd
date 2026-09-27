import argparse
import math
import sys
import time

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


def make_detection_pause(detector):
    '''
    Returns a should_pause function that is True while any PAUSE_LABELS are in view of the camera
    '''
    was_paused = False

    def should_pause():
        nonlocal was_paused
        if not detector.is_running():
            # Don't keep driving blind if the camera or model crashed
            raise RuntimeError(f"Object detection stopped: {detector.error!r}")
        paused = any(detector.seen_recently(label, DETECTION_HOLD_TIME) for label in PAUSE_LABELS)
        if paused and not was_paused:
            # Log what triggered the pause, useful for spotting false detections
            seen = [f"{d['label']} {d['score']:.2f}" for d in detector.latest() if d['label'] in PAUSE_LABELS]
            print(f"DETECTED: {', '.join(seen) or 'recently seen ' + '/'.join(PAUSE_LABELS)}")
        was_paused = paused
        return paused
    return should_pause


def wait_while_paused(should_pause):
    if should_pause is None or not should_pause():
        return
    print("PAUSED: waiting for path to clear")
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
    state_to_symbol = {
        GridState.UNKNOWN: "?",
        GridState.CLEAR: ".",
        GridState.OCCUPIED: "X",
    }
    print("UNKNOWN is ? | CLEAR is . | OCCUPIED is X | CLEARANCE is + | PATH is * | GOAL is G | CAR is C")
    path_cells = set(path or [])
    car_cell = (map_array.car_origin_row, map_array.car_origin_column)
    for row in range(map_array.grid.shape[0]):
        row_characters = []
        for column in range(map_array.grid.shape[1]):
            cell = (row, column)
            state = map_array.get_grid_state(row, column)
            if cell == car_cell:
                row_characters.append("C")
            elif cell == goal_cell:
                row_characters.append("G")
            elif cell in path_cells:
                row_characters.append("*")
            elif blocked[row][column] and state != GridState.OCCUPIED:
                row_characters.append("+")
            else:
                row_characters.append(state_to_symbol[state]) # type: ignore
        print("".join(row_characters))


def navigate(goal_forward, goal_right, servo_offset, should_pause=None):
    '''
    Drives to the goal at (goal_forward, goal_right) cm relative to the car's starting pose by
    repeatedly scanning, planning a route with A*, and following part of it.

    `should_pause` is an optional function that returns True while the car should wait in place
    (e.g. while object detection sees a person).
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
            # Imported here so navigation can run without PyTorch/camera when --no-detection is used
            from .background_detector import BackgroundDetector
            print("Starting object detection...")
            detector = BackgroundDetector(min_score=DETECTION_MIN_SCORE).start()
            should_pause = make_detection_pause(detector)
        navigate(args.forward, args.right, args.servo_offset, should_pause)
    except KeyboardInterrupt:
        print("\nStopping")
    finally:
        fc.stop()
        fc.servo.set_angle(0)
        if detector is not None:
            detector.stop()


if __name__ == "__main__":
    main()

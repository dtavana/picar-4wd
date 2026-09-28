import heapq
from itertools import count
import numpy as np

# Local grid directions as (row delta, column delta), indexed so that
# (direction + 1) % 4 is a 90 degree right turn from direction
FORWARD, RIGHT, BACKWARD, LEFT = 0, 1, 2, 3
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Extra cost (in cells) added for every 90 degree turn in a path
TURN_COST = 20

def inflate_obstacles(occupied, radius):
    """
    Returns a copy of the boolean "occupied" grid where every cell within "radius" cells
    of an occupied cell is also marked, giving obstacles clearance for the car's body
    """
    inflated = occupied.copy()
    rows, columns = occupied.shape
    for row_delta in range(-radius, radius + 1):
        for column_delta in range(-radius, radius + 1):
            if row_delta * row_delta + column_delta * column_delta > radius * radius:
                continue
            # Shifting the occupied grid by (row_delta, column_delta) and merging it in
            inflated[
                max(row_delta, 0):rows + min(row_delta, 0),
                max(column_delta, 0):columns + min(column_delta, 0),
            ] |= occupied[
                max(-row_delta, 0):rows + min(-row_delta, 0),
                max(-column_delta, 0):columns + min(-column_delta, 0),
            ]
    return inflated

def nearest_free_cell(blocked, target):
    """
    Returns "target" if it is free, otherwise the free cell closest to it (Manhattan distance),
    or None if every cell is blocked
    """
    if not blocked[target]:
        return target
    free_cells = np.argwhere(~blocked)
    if len(free_cells) == 0:
        return None
    distances = np.abs(free_cells - np.array(target)).sum(axis=1)
    row, column = free_cells[np.argmin(distances)]
    return int(row), int(column)


def find_path(blocked, start, goal, start_direction=FORWARD):
    """
    Runs A* over the 4-connected grid from "start" to "goal", returning the list of
    (row, column) cells on the path (including start and goal), or None if unreachable.

    The search state includes the direction the car is facing so that turns can be
    penalized with TURN_COST. The start cell is never treated as blocked.
    """
    rows, columns = blocked.shape
    blocked_rows = blocked.tolist()

    def heuristic(row, column):
        return abs(row - goal[0]) + abs(column - goal[1])

    start_state = (start[0], start[1], start_direction)
    cost_so_far = {start_state: 0}
    came_from = {start_state: None}
    closed = set()
    # Counter breaks ties between equal entries so states are never compared
    tie_breaker = count()
    start_heuristic = heuristic(*start)
    open_heap = [(start_heuristic, start_heuristic, next(tie_breaker), start_state)]

    while open_heap:
        _, _, _, state = heapq.heappop(open_heap)
        if state in closed:
            continue
        closed.add(state)
        row, column, direction = state
        if (row, column) == goal:
            return reconstruct_path(came_from, state)
        for next_direction, (row_delta, column_delta) in enumerate(MOVES):
            next_row, next_column = row + row_delta, column + column_delta
            if not (0 <= next_row < rows and 0 <= next_column < columns):
                continue
            if blocked_rows[next_row][next_column]:
                continue
            quarter_turns = (next_direction - direction) % 4
            turns = min(quarter_turns, 4 - quarter_turns)
            next_cost = cost_so_far[state] + 1 + turns * TURN_COST
            next_state = (next_row, next_column, next_direction)
            if next_cost < cost_so_far.get(next_state, float("inf")):
                cost_so_far[next_state] = next_cost
                came_from[next_state] = state
                next_heuristic = heuristic(next_row, next_column)
                heapq.heappush(
                    open_heap,
                    (next_cost + next_heuristic, next_heuristic, next(tie_breaker), next_state),
                )
    return None

def reconstruct_path(came_from, state):
    path = []
    while state is not None:
        path.append((state[0], state[1]))
        state = came_from[state]
    path.reverse()
    return path

def path_to_segments(path):
    # Compressing a list of adjacent cells into straight segments of (direction, cell_count)
    segments = []
    for (row, column), (next_row, next_column) in zip(path, path[1:]):
        direction = MOVES.index((next_row - row, next_column - column))
        if segments and segments[-1][0] == direction:
            segments[-1] = (direction, segments[-1][1] + 1)
        else:
            segments.append((direction, 1))
    return segments

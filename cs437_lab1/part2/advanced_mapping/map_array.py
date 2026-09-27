from enum import IntEnum
from math import cos, hypot, radians, sin

import numpy as np

# Width of the underlying 2D numpy array
WIDTH = 150
# Height of the underlying 2D numpy array
HEIGHT = 150
# Size in cm of each cell in the underlying 2D numpy array
CELL_SIZE = 1

# Maximum separation in cm between neighboring obstacle coordinates
# that can be considered a continous surface
MAX_COORDINATE_DISTANCE_DELTA = 15


class GridState(IntEnum):
    UNKNOWN = -1
    CLEAR = 0
    OCCUPIED = 1


class MapArray:
    """
    Implements a 2D numpy array to reflect obstacles from the car's perspective.
    The car's origin is defined as the bottom middle of the array:

    X X X X X
    X X X X X
    X X C X X

    Therefore with a CELL_SIZE of 1cm, an obstacle 2cm ahead in
    the grid above would be represented as updating the 1st row, 3rd column [0, 2]
    """

    def __init__(self):
        self.grid = np.empty(
            (int(HEIGHT / CELL_SIZE), int(WIDTH / CELL_SIZE)), dtype=np.int8
        )
        # Origin should be the bottom middle of the grid, therefore the x
        # coordinate of the origin is half the number of columns in the grid
        # and the y coordinate of the origin is the total number of rows - 1
        self.car_origin_column = int(self.grid.shape[1] / 2)
        self.car_origin_row = self.grid.shape[0] - 1
        self.reset()

    def reset(self):
        """
        Resets the grid to mark all cells as unknown
        """
        self.grid.fill(GridState.UNKNOWN)

    def update_grid(self, row, column, state):
        self.grid[row][column] = state

    def get_grid_state(self, row, column):
        return self.grid[row][column]

    def angle_distance_to_cell(self, angle, distance):
        """
        Converts a given servo angle and distance into coordinates usable in the grid array
        """
        # Get obstacle coordinates relative to the car in centimeters using trigonometry
        angle_rad = radians(-angle)
        x_distance, y_distance = distance * sin(angle_rad), distance * cos(angle_rad)
        # Translate distance coordiante to grid array coordinates
        column = self.car_origin_column + round(x_distance / CELL_SIZE)
        row = self.car_origin_row - round(y_distance / CELL_SIZE)
        # Return row, column as numpy indexes array in that order
        return row, column

    def in_bounds(self, row, column):
        row_ok = 0 <= row < self.grid.shape[0]
        column_ok = 0 <= column < self.grid.shape[1]
        return row_ok and column_ok

    def valid_connection(self, first_scan, second_scan):
        first_angle, first_distance = (
            first_scan["position"].angle,
            first_scan["distance"],
        )
        second_angle, second_distance = (
            second_scan["position"].angle,
            second_scan["distance"],
        )
        if first_distance is GridState.UNKNOWN or second_distance is GridState.UNKNOWN:
            # At least one of the scan distances were invalid, return early
            return False
        first_row, first_column = self.angle_distance_to_cell(
            first_angle, first_distance
        )
        second_row, second_column = self.angle_distance_to_cell(
            second_angle, second_distance
        )
        distance_delta = (
            hypot(second_row - first_row, second_column - first_column) * CELL_SIZE
        )
        return distance_delta <= MAX_COORDINATE_DISTANCE_DELTA

    def connected_cells(self, start_row, start_column, end_row, end_column):
        """
        Returns all cells on the line between [(start_row, start_column), (end_row, end_column)]
        """
        cells = []
        row_delta, column_delta = end_row - start_row, end_column - start_column
        # The total number of cells we need to update is the maximum of the row delta or column delta
        num_cells = max(abs(row_delta), abs(column_delta))
        if num_cells == 0:
            return cells
        for cell in range(num_cells + 1):
            # We need to calculate how far along the line we are to know which intermediate cells to alter
            line_progress = cell / num_cells
            cells.append(
                (
                    round(start_row + row_delta * line_progress),
                    round(start_column + column_delta * line_progress),
                )
            )
        return cells

    def fill_clear_cells(self, all_connected_cells):
        for connected_cells in all_connected_cells:
            for connected_row, connected_column in connected_cells:
                target_cells = self.connected_cells(
                    self.car_origin_row,
                    self.car_origin_column,
                    connected_row,
                    connected_column,
                )
                for row, column in target_cells:
                    if not self.in_bounds(row, column):
                        break
                    state = self.get_grid_state(row, column)
                    if state == GridState.OCCUPIED:
                        break
                    if self.get_grid_state(row, column) == GridState.UNKNOWN:
                        self.update_grid(row, column, GridState.CLEAR)

    def update_from_scan(self, scan_results):
        self.reset()
        prev_scan = None
        all_connected_cells = []
        for scan in scan_results:
            position, distance = scan["position"], scan["distance"]
            if distance is GridState.UNKNOWN:
                # We got an invalid distance, because the grid is initialzied with everything unknown no work to do here
                # other than breaking the previous scan chain to not interpolate occupied cells
                prev_scan = None
                continue
            row, column = self.angle_distance_to_cell(position.angle, distance)
            if self.in_bounds(row, column):
                self.update_grid(row, column, GridState.OCCUPIED)
            if prev_scan is not None and self.valid_connection(prev_scan, scan):
                # We can interpolate intermediate occupied cells here
                prev_row, prev_column = self.angle_distance_to_cell(
                    prev_scan["position"].angle, prev_scan["distance"]
                )
                connecting_cells = self.connected_cells(
                    prev_row, prev_column, row, column
                )
                all_connected_cells.append(connecting_cells)
                for connecting_row, connecting_column in connecting_cells:
                    if self.in_bounds(connecting_row, connecting_column):
                        self.update_grid(
                            connecting_row, connecting_column, GridState.OCCUPIED
                        )
            prev_scan = scan
        self.fill_clear_cells(all_connected_cells)

    def visualize(self, block_size=3):
        state_to_symbol = {
            GridState.UNKNOWN: "?",
            GridState.CLEAR: ".",
            GridState.OCCUPIED: "X",
        }
        symbol_text = [f"{x.name} is {y}" for x, y in state_to_symbol.items()]
        symbol_text.append("CAR is C")
        print(" | ".join(symbol_text))
        # Skip completely unknown blocks of rows, stopping before the car.
        start_row = 0
        while start_row + block_size <= self.car_origin_row:
            # Scan every block_size rows to see if any are unknown
            if np.any(
                self.grid[start_row : start_row + block_size] != GridState.UNKNOWN
            ):
                # If all cells in this subsection of rows is UNKNOWN, terminate output here
                break
            start_row += block_size
        # Keep one empty display row as a margin above the map.
        start_row = max(0, start_row - block_size)
        for row in range(start_row, self.grid.shape[0], block_size):
            row_characters = []
            for column in range(0, self.grid.shape[1], block_size):
                if (
                    row <= self.car_origin_row < row + block_size
                    and column <= self.car_origin_column < column + block_size
                ):
                    row_characters.append("C")
                else:
                    block = self.grid[
                        row : row + block_size, column : column + block_size
                    ]
                    if np.any(block == GridState.OCCUPIED):
                        state = GridState.OCCUPIED
                    elif np.all(block == GridState.CLEAR):
                        state = GridState.CLEAR
                    else:
                        state = GridState.UNKNOWN
                    row_characters.append(state_to_symbol[state])
            print("".join(row_characters))

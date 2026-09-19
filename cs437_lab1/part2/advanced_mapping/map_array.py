import numpy as np
from enum import IntEnum
from math import sin, cos, radians

# Width of the underlying 2D numpy array
WIDTH = 100
# Height of the underlying 2D numpy array
HEIGHT = 100
# Size in cm of each cell in the underlying 2D numpy array
CELL_SIZE = 1

class GridState(IntEnum):
    UNKNOWN = -1
    CLEAR = 0
    OCCUPIED = 1
    
class MapArray:
    ''''
    Implements a 2D numpy array to reflect obstacles from the car's perspective.
    The car's origin is defined as the bottom middle of the array:

    X X X X X
    X X X X X
    X X C X X

    Therefore with a CELL_SIZE of 1cm, an obstacle 2cm ahead in 
    the grid above would be represented as updating the 1st row, 3rd column [0, 2]
    '''
    def __init__(self):
        self.grid = np.empty((int(HEIGHT / CELL_SIZE), int(WIDTH / CELL_SIZE)), dtype=np.int8)
        # Origin should be the bottom middle of the grid, therefore the x
        # coordinate of the origin is half the number of columns in the grid
        # and the y coordinate of the origin is the total number of rows - 1
        self.car_origin_column = self.grid.shape[1] // 2
        self.car_origin_row = self.grid.shape[0] - 1
        self.reset()

    def reset(self):
        '''
        Resets the grid to mark all cells as unknown
        '''
        self.grid.fill(GridState.UNKNOWN)

    def angle_distance_to_cell(self, angle, distance):
        '''
        Converts a given servo angle and distance into coordinates usable in the grid array
        '''
        # Get obstacle coordinates relative to the car in centimeters using trigonometry
        angle_rad = radians(angle)
        x_distance, y_distance = distance * sin(angle_rad), distance * cos(angle_rad)
        # Translate distance coordiante to grid array coordinates
        column = self.car_origin_column + int(round(x_distance / CELL_SIZE))
        row = self.car_origin_row - int(round(y_distance / CELL_SIZE))
        # Return row, column as numpy indexes array in that order
        return row, column

    def in_bounds(self, row, column):
        row_ok = 0 <= row < self.grid.shape[0]
        column_ok = 0 <= column < self.grid.shape[1]
        return row_ok and column_ok

    def valid_connection(self, first_scan, second_scan):
        if first_scan.distance is GridState.UNKNOWN or second_scan.distance is GridState.UNKNOWN:
            # At least one of the scan distances were invalid, return early
            return

    def update_from_scan(self, scan_results):
        self.reset()
        for scan in scan_results:
            position, distance = scan["position"], scan["distance"]
            if distance is GridState.UNKNOWN:
                # We got an invalid distance, because the grid is initialzied with everything unknown no work to do here
                continue
            row, column = self.angle_distance_to_cell(position.angle, distance)
            if self.in_bounds(row, column):
                self.grid[row][column] = GridState.OCCUPIED
        
        

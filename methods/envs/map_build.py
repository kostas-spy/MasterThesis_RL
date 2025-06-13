import pygame
import random
import numpy as np
import os
import config as c

'''
Map: Saves or loads map, checks walls and obstacles around and generate map (commented)
    Functions: save_map, load_map, is_open, is_within_bounds, is_portal, get_portal
'''


class Map:

    # COMPASS
    COMPASS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}

    # Map variables init and load or generate new map
    def __init__(self, map_walls=None, map_obstacles=None):

        # Map member variables
        self.map_walls = map_walls
        self.map_obstacles = map_obstacles

        # If map from file then get its size
        if self.map_walls is not None and self.map_obstacles is not None:
            if isinstance(self.map_walls, (np.ndarray, np.generic)) and len(self.map_walls.shape) == 2:
                self.map_size = tuple(self.map_walls.shape)
            else:
                raise ValueError("maps must be a 2D NumPy array.")
        else:
            raise ValueError("map_walls and map_obst must exist")

    # Save map
    def save_map(self, file_path, file):
        np.save(file_path, file, allow_pickle=False, fix_imports=True)

    # Load map
    @classmethod
    def load_map(cls, file_path):
        return np.load(file_path, allow_pickle=False, fix_imports=True)

    # check if cell is within bounds and if it is, if the wall is open
    def is_open(self, cell_id, dir):
        # check if it would be out-of-bound
        x1 = cell_id[0] + self.COMPASS[dir][0]
        y1 = cell_id[1] + self.COMPASS[dir][1]

        # if cell is still within bounds after the move
        if self.is_within_bound(x1, y1):
            # check if the wall is opened
            this_wall = bool(self.get_walls_status(self.map_walls[cell_id[0], cell_id[1]])[dir])
            other_wall = bool(self.get_walls_status(self.map_walls[x1, y1])[self.__get_opposite_wall(dir)])
            # check if the obstacle doesn't exists
            other_obst = bool(self.map_obstacles[x1, y1])
            return (this_wall or other_wall) and not other_obst
        return False

    def is_within_bound(self, x, y):
        # true if cell is still within bounds after the move
        return 0 <= x < self.MAP_W and 0 <= y < self.MAP_H

    @property
    def MAP_W(self):
        return int(self.map_size[0])

    @property
    def MAP_H(self):
        return int(self.map_size[1])

    @classmethod
    def get_walls_status(cls, cell):
        walls = {
            "N": (cell & 0x1) >> 0,
            "E": (cell & 0x2) >> 1,
            "S": (cell & 0x4) >> 2,
            "W": (cell & 0x8) >> 3,
        }
        return walls

    @classmethod
    def __get_opposite_wall(cls, dirs):
        opposite_dirs = ""
        for dir in dirs:
            if dir == "N":
                opposite_dir = "S"
            elif dir == "S":
                opposite_dir = "N"
            elif dir == "E":
                opposite_dir = "W"
            elif dir == "W":
                opposite_dir = "E"
            else:
                raise ValueError("The only valid directions are (N, S, E, W).")
            opposite_dirs += opposite_dir

        return opposite_dirs


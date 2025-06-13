import pygame
import random
import numpy as np
import os
from methods.envs.map_build import Map

path = os.getcwd()
dir_name = path + '\\map_samples\\'

'''
Appendix: 
u-N, r-E, l-W, d-S
No lines: 15
All lines: 0
1 line: 7(l), 11(d), 13(r), 14(u)
Corners: 3(dl), 6(ul), 9(dr), 12(ur)
Across: 5(lr), 10(ud)
3 lines (ONLY MISSING): 1(u), 2(r), 4(d), 8(l) 
'''

data_wall = np.array([[6, 14, 12],
                      [7, 15, 13],
                      [3, 11, 9]])

data_obst = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]])

# Name
name = 'map_3x3'
map_dir = os.path.join(dir_name, name)
if not os.path.exists(map_dir):
    os.mkdir(map_dir)
map_path_walls = os.path.join(map_dir, name + "_walls.npy")
map_path_obst = os.path.join(map_dir, name + "_obst.npy")

data_wall = data_wall.transpose()
data_obst = data_obst.transpose()
print(map_path_walls)
data = np.array([data_wall, data_obst])
print(data)

map = Map(map_walls=data_wall, map_obstacles=data_obst)

# Save map_cells and map_obstacles
map.save_map(file_path=map_path_walls, file=data_wall)
map.save_map(file_path=map_path_obst, file=data_obst)

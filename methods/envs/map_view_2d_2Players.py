import pygame
import random
import numpy as np
import os
from methods.envs.map_build import Map
import config as c


class MapView2D:

    def __init__(self, map_name="CTF2D", map_file_path_wall=None, map_file_path_obst=None,
                 screen_size=c.SCREEN_SIZE, enable_render=True, problem=None):
        pygame.init()
        pygame.display.set_caption(map_name)
        # self.clock = pygame.time.Clock()
        self.__game_over = False
        self.__enable_render = enable_render
        self.problem = problem

        if map_file_path_wall is not None:
            self.__map = Map(map_walls=Map.load_map(map_file_path_wall),
                               map_obstacles=Map.load_map(map_file_path_obst))

        self.map_size = self.__map.map_size
        if self.__enable_render is True:
            self.screen = pygame.display.set_mode(screen_size)
            self.__screen_size = tuple(map(sum, zip(screen_size, (-1, -1))))

        self.__entrance_c = np.zeros(2, dtype=int)
        self.__entrance_r = np.array(self.map_size) - np.ones(2, dtype=int)
        self.__robot_c = self.__entrance_c
        self.__robot_r = self.__entrance_r

        self.__obstacles = Map.load_map(map_file_path_obst)
        self.obst_indexes = np.where(self.__obstacles == 1)
        self.obst_indexes = [list(i) for i in self.obst_indexes]

        self.obst_ind = []
        for w in zip(self.obst_indexes[0], self.obst_indexes[1]):
            self.obst_ind.append(w)
        while True:
            # Set the Goal (upper-left corner)
            rand_arr = np.array((random.randint(1, self.map_size[0]), random.randint(1, self.map_size[0])))
            self.__robot_r = np.array(self.map_size) - rand_arr
            if tuple(self.__robot_r) in self.obst_ind:
                pass
            else:
                break

        if self.__enable_render is True:
            self.background = pygame.Surface(self.screen.get_size()).convert()
            self.background.fill((255, 255, 255))
            self.map_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
            self.map_layer.fill((0, 0, 0, 0,))

            self.__draw_map()
            self.__draw_obstacles()
            self.__draw_robot_c()
            self.__draw_robot_r()
            self.__draw_entrance_c()
            self.__draw_entrance_r()

    # Updates and checks for quit
    def update(self, mode="human"):
        try:
            img_output = self.__view_update(mode)
            self.__controller_update()
        except Exception as e:
            self.__game_over = True
            self.quit_game()
            raise e
        else:
            return img_output

    # Quits game
    def quit_game(self):
        try:
            self.__game_over = True
            if self.__enable_render is True:
                pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

    # Chaser movement
    def move_robot_c(self, dir):
        if dir not in self.__map.COMPASS.keys():
            raise ValueError("dir cannot be %s. The only valid dirs are %s."
                             % (str(dir), str(self.__map.COMPASS.keys())))

        if self.__map.is_open(self.__robot_c, dir):
            self.__draw_robot_c(transparency=0)
            self.__robot_c += np.array(self.__map.COMPASS[dir])
            self.__draw_robot_c(transparency=255)
            return True
        return False

    # Runner movement
    def move_robot_r(self, dir):
        if dir not in self.__map.COMPASS.keys():
            raise ValueError("dir cannot be %s. The only valid dirs are %s."
                             % (str(dir), str(self.__map.COMPASS.keys())))

        if self.__map.is_open(self.__robot_r, dir):
            self.__draw_robot_r(transparency=0)
            self.__robot_r += np.array(self.__map.COMPASS[dir])
            self.__draw_robot_r(transparency=255)
            return True
        return False

    def reset_robot_c(self):
        self.__draw_robot_c(transparency=0)
        while True:
            rand_arr = np.array((random.randint(1, self.map_size[0]), random.randint(1, self.map_size[0])))
            self.__robot_c = np.array(self.map_size) - rand_arr
            if tuple(self.__robot_c) in self.obst_ind:
                pass
            else:
                if tuple(self.__robot_c) == tuple(self.__robot_r):
                    pass
                else:
                    break
        self.__draw_robot_c(transparency=255)

    def reset_robot_r(self):
        self.__draw_robot_r(transparency=0)
        while True:
            rand_arr = np.array((random.randint(1, self.map_size[0]), random.randint(1, self.map_size[0])))
            self.__robot_r = np.array(self.map_size) - rand_arr
            if tuple(self.__robot_r) in self.obst_ind:
                pass
            else:
                break

        # self.__robot_r = np.array(self.map_size) - np.ones(2, dtype=int)
        self.__draw_robot_r(transparency=255)

    # Checks for quit
    def __controller_update(self):
        if not self.__game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.__game_over = True
                    self.quit_game()

    # Redraws everything and blit/flip
    def __view_update(self, mode="human"):
        if not self.__game_over:
            # update the robot's position
            self.__draw_entrance_c()
            self.__draw_entrance_r()
            # self.__draw_goal()
            self.__draw_obstacles()
            self.__draw_robot_c()
            self.__draw_robot_r()

            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.map_layer, (0, 0))

            if mode == "human":
                pygame.display.flip()

            return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))

    # Draw horizontal, vertical lines and gets cells' walls info
    def __draw_map(self):
        if self.__enable_render is False:
            return

        line_colour = (0, 0, 0, 255)

        # drawing the horizontal lines
        for y in range(self.map.MAP_H + 1):
            pygame.draw.line(self.map_layer, line_colour, (0, y * self.CELL_H),
                             (self.SCREEN_W, y * self.CELL_H))

        # drawing the vertical lines
        for x in range(self.map.MAP_W + 1):
            pygame.draw.line(self.map_layer, line_colour, (x * self.CELL_W, 0),
                             (x * self.CELL_W, self.SCREEN_H))

        # breaking the walls
        for x in range(len(self.map.map_walls)):
            for y in range(len(self.map.map_walls[x])):
                # check the which walls are open in each cell
                walls_status = self.map.get_walls_status(self.map.map_walls[x, y])
                dirs = ""
                for dir, open in walls_status.items():
                    if open:
                        dirs += dir
                self.__cover_walls(x, y, dirs)

    # Draw walls??
    def __cover_walls(self, x, y, dirs, colour=(0, 0, 255, 15)):
        if self.__enable_render is False:
            return

        dx = x * self.CELL_W
        dy = y * self.CELL_H

        if not isinstance(dirs, str):
            raise TypeError("dirs must be a str.")

        for dir in dirs:
            if dir == "S":
                line_head = (dx + 1, dy + self.CELL_H)
                line_tail = (dx + self.CELL_W - 1, dy + self.CELL_H)
            elif dir == "N":
                line_head = (dx + 1, dy)
                line_tail = (dx + self.CELL_W - 1, dy)
            elif dir == "W":
                line_head = (dx, dy + 1)
                line_tail = (dx, dy + self.CELL_H - 1)
            elif dir == "E":
                line_head = (dx + self.CELL_W, dy + 1)
                line_tail = (dx + self.CELL_W, dy + self.CELL_H - 1)
            else:
                raise ValueError("The only valid directions are (N, S, E, W).")

            pygame.draw.line(self.map_layer, colour, line_head, line_tail)

    # Draw chaser
    def __draw_robot_c(self, colour=(0, 0, 255), transparency=255):
        if self.__enable_render is False:
            return

        x = int(self.__robot_c[0] * self.CELL_W + self.CELL_W * 0.5 + 0.5)
        y = int(self.__robot_c[1] * self.CELL_H + self.CELL_H * 0.5 + 0.5)
        r = int(min(self.CELL_W, self.CELL_H) / 5 + 0.5)

        pygame.draw.circle(self.map_layer, colour + (transparency,), (x, y), r)
        pygame.draw.circle(self.map_layer, (0, 0, 0) + (transparency,), (x, y), r, width=5)

    # Draw runner
    def __draw_robot_r(self, colour=(255, 0, 0), transparency=255):
        if self.__enable_render is False:
            return

        x = int(self.__robot_r[0] * self.CELL_W + self.CELL_W * 0.5 + 0.5)
        y = int(self.__robot_r[1] * self.CELL_H + self.CELL_H * 0.5 + 0.5)
        r = int(min(self.CELL_W, self.CELL_H) / 5 + 0.5)

        pygame.draw.circle(self.map_layer, colour + (transparency,), (x, y), r)
        pygame.draw.circle(self.map_layer, (0, 0, 0) + (transparency,), (x, y), r, width=2)

    # Draw chaser base
    def __draw_entrance_c(self, colour=(0, 0, 150), transparency=235):
        self.__colour_cell(self.__entrance_c, colour=colour, transparency=transparency)

    # Draw runner base
    def __draw_entrance_r(self, colour=(150, 0, 0), transparency=235):
        self.__colour_cell(self.__entrance_r, colour=colour, transparency=transparency)

    # Draw obstacles light gray
    def __draw_obstacles(self, colour=(211, 211, 211), transparency=235):
        obstacles_to_draw = zip(*np.where(self.__obstacles == 1))
        for i in obstacles_to_draw:
            self.__colour_cell(i, colour=colour, transparency=transparency)

    # Draw cells with specific color (called from __draw_entrance, __draw_goal, __draw_portals)
    def __colour_cell(self, cell, colour, transparency):
        if self.__enable_render is False:
            return

        if not (isinstance(cell, (list, tuple, np.ndarray)) and len(cell) == 2):
            raise TypeError("cell must a be a tuple, list, or numpy array of size 2")

        x = int(cell[0] * self.CELL_W + 0.5 + 1)
        y = int(cell[1] * self.CELL_H + 0.5 + 1)
        w = int(self.CELL_W + 0.5 - 1)
        h = int(self.CELL_H + 0.5 - 1)
        pygame.draw.rect(self.map_layer, colour + (transparency,), (x, y, w, h))


    @property
    def map(self):
        return self.__map

    @property
    def robot_c(self):
        return self.__robot_c

    @property
    def robot_r(self):
        return self.__robot_r

    @property
    def entrance_c(self):
        return self.__entrance_c

    @property
    def entrance_r(self):
        return self.__entrance_r

    @property
    def obstacles(self):
        return self.__obstacles

    @property
    def game_over(self):
        return self.__game_over

    @property
    def SCREEN_SIZE(self):
        return tuple(self.__screen_size)

    @property
    def SCREEN_W(self):
        # <- ->
        return int(self.SCREEN_SIZE[0])

    @property
    def SCREEN_H(self):
        # ^ v
        return int(self.SCREEN_SIZE[1])

    @property
    def CELL_W(self):
        return float(self.SCREEN_W) / float(self.map.MAP_W)

    @property
    def CELL_H(self):
        return float(self.SCREEN_H) / float(self.map.MAP_H)

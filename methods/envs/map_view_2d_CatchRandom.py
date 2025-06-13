import pygame
import random
import numpy as np
import os
from methods.envs.map_build import Map
import config as c

'''
MapEnv: Renders an instance using pygame (calls Map from map_build)
    Draws the map (robot, entrance, goal, portals)
    Functions: update, quit_game, move_robot, reset_robot, __controller_update, __view_update, 
               __draw_map, __cover_walls, __draw_robot, __draw_entrance, __draw_goal, __draw_portals, __draw_obstacles, __colour_cell
'''


class MapView2D:

    # Setups everything, pygame Configs, load or create map, entrance-goal-robot positions,
    # draw backgrounds and __draw_map, __draw_robot, __draw_entrance, __draw_goal, __draw_portals
    def __init__(self, map_name="CTF2D", map_file_path_wall=None, map_file_path_obst=None,
                 screen_size=c.SCREEN_SIZE, enable_render=True, problem=None):
        # PyGame configurations
        pygame.init()
        pygame.display.set_caption(map_name)
        # self.clock = pygame.time.Clock()
        self.__game_over = False
        self.__enable_render = enable_render
        self.problem = problem

        # Creates a map with map_build.py or loads a map from map_file_path
        if map_file_path_wall is not None:
            self.__map = Map(map_walls=Map.load_map(map_file_path_wall),
                               map_obstacles=Map.load_map(map_file_path_obst))

        self.map_size = self.__map.map_size
        if self.__enable_render is True:
            # to show the right and bottom border (DISPLAY)
            self.screen = pygame.display.set_mode(screen_size)
            self.__screen_size = tuple(map(sum, zip(screen_size, (-1, -1))))

        # Set the starting point (down-right corner)
        self.__entrance = np.zeros(2, dtype=int)

        # Set obstacles array (load from map_file_path_obst)
        self.__obstacles = Map.load_map(map_file_path_obst)
        self.obst_indexes = np.where(self.__obstacles == 1)
        self.obst_indexes = [list(i) for i in self.obst_indexes]

        self.obst_ind = []
        for w in zip(self.obst_indexes[0], self.obst_indexes[1]):
            self.obst_ind.append(w)
        while True:
            # Set the Goal (upper-left corner)
            rand_arr = np.array((random.randint(1, self.map_size[0]), random.randint(1, self.map_size[0])))
            self.__goal = np.array(self.map_size) - rand_arr
            if tuple(self.__goal) in self.obst_ind:
                pass
            else:
                break

        # Create the Robot
        self.__robot = self.entrance

        # Create random bot enemies (if applied)
        self.__bot_enemy = self.__goal

        # Create map (background, surface, walls, portals, robot, entrance, goal)
        if self.__enable_render is True:
            # Create a background
            self.background = pygame.Surface(self.screen.get_size()).convert()
            self.background.fill((255, 255, 255))

            # Create a layer for the map
            self.map_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
            self.map_layer.fill((0, 0, 0, 0,))

            # show the map walls
            self.__draw_map()

            # show the map obstacles
            self.__draw_obstacles()

            # show the robot
            self.__draw_robot()

            # show the entrance
            self.__draw_entrance()

            # show the goal
            # self.__draw_goal()

            self.__draw_bot_enemy()

    # Updates and checks for quit
    def update(self, mode="human"):
        try:
            # Updates everything and blits/flips
            img_output = self.__view_update(mode)
            # Checks for quit_game
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

    # Robot movement and draw its update
    def move_robot(self, dir):
        if dir not in self.__map.COMPASS.keys():
            raise ValueError("dir cannot be %s. The only valid dirs are %s."
                             % (str(dir), str(self.__map.COMPASS.keys())))

        if self.__map.is_open(self.__robot, dir):
            # update the drawing
            self.__draw_robot(transparency=0)
            # move the robot
            self.__robot += np.array(self.__map.COMPASS[dir])
            self.__draw_robot(transparency=255)
            return True
        return False

    def move_enemy_bot(self, dir):
        if dir not in self.__map.COMPASS.keys():
            raise ValueError("dir cannot be %s. The only valid dirs are %s."
                             % (str(dir), str(self.__map.COMPASS.keys())))

        if self.__map.is_open(self.__bot_enemy, dir):
            # update the drawing
            self.__draw_bot_enemy(transparency=0)
            # move the robot
            self.__bot_enemy += np.array(self.__map.COMPASS[dir])
            self.__draw_bot_enemy(transparency=255)
            return True
        return False

    # Resets robot to the initial position (0, 0)
    def reset_robot(self):
        # Delete
        self.__draw_robot(transparency=0)
        # Return to (0, 0)
        self.__robot = np.zeros(2, dtype=int)
        # Redraw
        self.__draw_robot(transparency=255)

    # Resets robot to the initial position (0, 0)
    def reset_bot_enemy(self):
        # Delete
        self.__draw_bot_enemy(transparency=0)
        # Return t random position
        while True:
            rand_arr = np.array((random.randint(1, self.map_size[0]), random.randint(1, self.map_size[0])))
            self.__goal = np.array(self.map_size) - rand_arr
            if tuple(self.__goal) in self.obst_ind:
                pass
            else:
                break
        self.__bot_enemy = self.__goal
        # Redraw
        self.__draw_bot_enemy(transparency=255)

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
            self.__draw_entrance()
            # self.__draw_goal()
            self.__draw_obstacles()
            self.__draw_robot()
            if self.problem == 'GoalAndBackAvoidEnemies':
                self.__draw_bot_enemy()

            # update the screen
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.map_layer, (0, 0))

            if mode == "human":
                pygame.display.flip()

            return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))
            # return np.flipud(pygame.surfarray.array3d(pygame.display.get_surface()))

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

    # Draw robot
    def __draw_robot(self, colour=(0, 0, 255), transparency=255):
        if self.__enable_render is False:
            return

        x = int(self.__robot[0] * self.CELL_W + self.CELL_W * 0.5 + 0.5)
        y = int(self.__robot[1] * self.CELL_H + self.CELL_H * 0.5 + 0.5)
        r = int(min(self.CELL_W, self.CELL_H) / 5 + 0.5)

        pygame.draw.circle(self.map_layer, colour + (transparency,), (x, y), r)
        pygame.draw.circle(self.map_layer, (0, 0, 0) + (transparency,), (x, y), r, width=5)

    # Draw entrance blue
    def __draw_entrance(self, colour=(0, 0, 150), transparency=235):
        self.__colour_cell(self.entrance, colour=colour, transparency=transparency)

    # Draw goal red
    def __draw_goal(self, colour=(150, 0, 0), transparency=235):
        self.__colour_cell(self.goal, colour=colour, transparency=transparency)

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

    # Draw bot random movement enemies
    def __draw_bot_enemy(self, colour=(255, 0, 0), transparency=255):
        if self.__enable_render is False:
            return

        x = int(self.__bot_enemy[0] * self.CELL_W + self.CELL_W * 0.5 + 0.5)
        y = int(self.__bot_enemy[1] * self.CELL_H + self.CELL_H * 0.5 + 0.5)
        r = int(min(self.CELL_W, self.CELL_H) / 5 + 0.5)

        pygame.draw.circle(self.map_layer, colour + (transparency,), (x, y), r)
        pygame.draw.circle(self.map_layer, (0, 0, 0) + (transparency,), (x, y), r, width=5)

    @property
    def map(self):
        return self.__map

    @property
    def robot(self):
        return self.__robot

    @property
    def bot_enemy(self):
        return self.__bot_enemy

    @property
    def entrance(self):
        return self.__entrance

    @property
    def obstacles(self):
        return self.__obstacles

    @property
    def goal(self):
        return self.__goal

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

import numpy as np
import pandas as pd
import os
import gym
import heapq
from gym import error, spaces, utils
from gym.utils import seeding
from methods.envs.map_view_2d_CatchStatic import MapView2D
import config as c


class MapEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], }
    ACTION = ['N', 'S', 'E', 'W']

    def __init__(self, map_name=None, map_file_wall=None, map_file_obst=None, enable_render=c.ENABLE_RENDER, problem=None):

        self.viewer = None
        self.enable_render = enable_render
        self.map_name = map_name
        self.display = None
        self.problem = problem

        # Load map_file from map_samples/*.npy
        if map_file_wall and map_file_obst:
            # View from MapView2D from map_view_2d_CatchStatic.py
            self.map_view = MapView2D(map_name=self.map_name, map_file_path_wall=map_file_wall, map_file_path_obst=map_file_obst, screen_size=c.SCREEN_SIZE,
                                        enable_render=enable_render, problem=self.problem)
        else:
            raise AttributeError("Must supply a map_file_wall path (str) and the map_file_obst path")

        self.map_size = self.map_view.map_size
        self.action_space = spaces.Discrete(2*len(self.map_size))

        low = np.zeros(len(self.map_size), dtype=int)
        low = np.append(low, np.zeros(len(self.map_size), dtype=int))
        high = np.array(self.map_size, dtype=int) - np.ones(len(self.map_size), dtype=int)
        high = np.append(high, np.array(self.map_size, dtype=int) - np.ones(len(self.map_size), dtype=int))

        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        print(self.action_space)
        print(self.observation_space)
        self.state = None
        self.steps_beyond_done = None
        self.bot_pos = self.map_view.bot_enemy

        self.seed()
        # self.reset()

    def __del__(self):
        if self.enable_render is True:
            self.map_view.quit_game()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = int(action)
        if isinstance(action, int):
            self.valid_mov = self.map_view.move_robot(self.ACTION[action])
        else:
            self.valid_mov = self.map_view.move_robot(action)

        reward, done, _ = self.reward_logic()

        self.state = self.map_view.robot
        self.bot_pos = self.map_view.bot_enemy
        self.state = np.append(self.state, self.bot_pos)
        info = {}

        return self.state, reward, done, info

    def reset(self):
        robot_init = self.map_view.reset_robot()
        enemy_init = self.map_view.reset_bot_enemy()
        self.state = np.array([robot_init[0], robot_init[1], enemy_init[0], enemy_init[1]])
        self.steps_beyond_done = None
        self.done = False
        return self.state

    def is_game_over(self):
        return self.map_view.game_over

    def render(self, mode="human", close=False):
        if close:
            self.map_view.quit_game()
        return self.map_view.update(mode)

    def reward_logic(self):
        dist = np.linalg.norm(self.map_view.robot - self.map_view.goal)
        if np.array_equal(self.map_view.robot, self.map_view.goal):
            reward = 100
            done = True
        else:
            if not self.valid_mov:
                reward = -5
            else:
                reward = -1
            done = False

        return reward, done, None

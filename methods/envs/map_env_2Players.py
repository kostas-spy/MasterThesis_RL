import numpy as np
import pandas as pd
import os
import random
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from methods.envs.map_view_2d_2Players import MapView2D
from keras.models import load_model
import config as c
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding, Reshape
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint, FileLogger, TrainEpisodeLogger, TrainIntervalLogger


class MapEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], }
    ACTION = ['N', 'S', 'E', 'W']

    # Class Initialization
    def __init__(self, map_name=None, map_file_wall=None, map_file_obst=None, enable_render=c.ENABLE_RENDER, problem=None, test_mode_1v1=0):

        self.viewer = None
        self.enable_render = enable_render
        self.map_name = map_name
        self.display = None
        self.problem = problem
        self.test_mode_1v1 = test_mode_1v1

        if map_file_wall and map_file_obst:
            self.map_view = MapView2D(map_name=self.map_name, map_file_path_wall=map_file_wall, map_file_path_obst=map_file_obst, screen_size=c.SCREEN_SIZE,
                                      enable_render=enable_render, problem=self.problem)
        else:
            raise AttributeError("Must supply a map_file_wall path (str) and the map_file_obst path")

        self.map_size = self.map_view.map_size
        self.action_space = spaces.Discrete(2 * len(self.map_size))

        low = np.zeros(len(self.map_size), dtype=int)
        low = np.append(low, np.zeros(len(self.map_size), dtype=int))
        high = np.array(self.map_size, dtype=int) - np.ones(len(self.map_size), dtype=int)
        high = np.append(high, np.array(self.map_size, dtype=int) - np.ones(len(self.map_size), dtype=int))

        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        print(self.action_space)
        print(self.observation_space)

        # initial condition
        self.state_c = None
        self.state_r = None
        self.steps_beyond_done = None
        self.c_turn = True  # Chaser plays first
        self.valid_mov_c = None
        self.valid_mov_r = None

        self.seed()
        self.reset()
        # self.reset_c()
        # self.reset_r()

        self.ccc = False

    # Finalizer function, occurs after quitting
    def __del__(self):
        if self.enable_render is True:
            self.map_view.quit_game()

    # Set random seed
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Step action
    def step(self, action):
        if self.test_mode_1v1:
            pass

        if self.c_turn:
            if np.issubdtype(type(action), np.integer):
                try:
                    self.valid_mov_c = self.map_view.move_robot_c(self.ACTION[int(action)])
                except:
                    print(action)
            else:
                self.valid_mov_c = self.map_view.move_robot_c(action)

            reward, done, _ = self.reward_logic()

            self.state_c = self.map_view.robot_c
            self.state_r = self.map_view.robot_r
            self.state_c = np.append(self.state_c, self.state_r)
            info = {}
            return self.state_c, reward, done, info

        if not self.c_turn:
            if not self.ccc:
                self.enemy_movement_path = 'C:\\Users\\Kostas\\Desktop\\Thesis\\Code\\Thesis_final\\methods\\dqn\\models\\weights\\DQN_2Players_Chaser_map_5x5_empty.h5f'
                memory = SequentialMemory(limit=50000, window_length=1)
                policy = EpsGreedyQPolicy(eps=0.2)
                self.test_model_c = DQNAgent(model=self.model_v1(), nb_actions=self.action_space.n, memory=memory, nb_steps_warmup=1000, target_model_update=1e-2, policy=policy, gamma=0.95)
                self.test_model_c.compile(Adam(lr=0.0001), metrics=['mse'])
                self.test_model_c.load_weights(filepath=self.enemy_movement_path)
                self.ccc = True

            if np.issubdtype(type(action), np.integer):
                try:
                    self.valid_mov_r = self.map_view.move_robot_r(self.ACTION[int(action)])
                except:
                    print(action)
            else:
                self.valid_mov_r = self.map_view.move_robot_r(action)

            reward, done, _ = self.reward_logic()

            enemy_action = self.test_model_c.forward(np.append(self.map_view.robot_c, self.map_view.robot_r))

            self.map_view.move_robot_c(self.ACTION[enemy_action])
            self.state_r = self.map_view.robot_r
            self.state_c = self.map_view.robot_c
            self.state_r = np.append(self.state_r, self.state_c)
            info = {}

            return self.state_r, reward, done, info

    # Reset position of agent
    def reset_c(self):
        self.map_view.reset_robot_c()
        self.state_c = np.zeros(4)
        self.steps_beyond_done = None
        self.done = False
        return self.state_c

    def reset_r(self):
        self.map_view.reset_robot_r()
        # self.state_r = np.array(self.map_size) - np.ones(2, dtype=int)
        self.state_r = np.zeros(4)
        self.steps_beyond_done = None
        self.done = False
        return self.state_r

    def reset(self):
        self.map_view.reset_robot_c()
        self.map_view.reset_robot_r()
        self.state_c = np.zeros(4)
        self.state_r = np.zeros(4)
        self.steps_beyond_done = None
        self.done = False
        if self.c_turn:
            return self.state_c
        else:
            return self.state_r

    # Game Over
    def is_game_over(self):
        return self.map_view.game_over

    # Render
    def render(self, mode="human", close=False):
        if close:
            self.map_view.quit_game()
        return self.map_view.update(mode)

    # Reward Logic
    def reward_logic(self):
        if self.c_turn:
            if np.array_equal(self.map_view.robot_r, self.map_view.robot_c):
                reward = 100
                done = True
            else:
                if not self.valid_mov_c:
                    reward = -5
                else:
                    reward = -1
                done = False
            return reward, done, None
        if not self.c_turn:
            if np.array_equal(self.map_view.robot_r, self.map_view.robot_c):
                reward = -100
                done = False
            else:
                if not self.valid_mov_r:
                    reward = 2
                else:
                    reward = 10
                done = False
            return reward, done, None

    def model_v1(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.observation_space.shape))
        model.add(Dense(50, activation='selu'))
        model.add(Dense(50, activation='selu'))
        model.add(Dense(50, activation='selu'))
        model.add(Dense(20, activation='selu'))
        model.add(Dense(self.action_space.n, activation='softmax'))
        # print(model.summary())
        return model
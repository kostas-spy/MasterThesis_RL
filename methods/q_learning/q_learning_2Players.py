import sys
import numpy as np
import math
import random
import os
import time
import gym
import methods


class QLearning:

    def __init__(self, env, name='ctf'):
        self.env = env
        self.name = name

        self.MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
        self.NUM_BUCKETS = self.MAZE_SIZE

        self.NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
        self.STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

        self.MIN_EXPLORE_RATE = 0.001
        self.MIN_LEARNING_RATE = 0.2
        self.DECAY_FACTOR = np.prod(self.MAZE_SIZE, dtype=float) / 10.0

        self.NUM_EPISODES = 200000
        self.MAX_T = self.MAZE_SIZE[0] * 4
        self.STREAK_TO_END = 100
        self.SOLVED_T = self.MAX_T // 2
        self.RENDER_MAZE = True
        self.end_list = [0, 0]

        self.q_table_c = np.zeros(self.NUM_BUCKETS + (self.NUM_ACTIONS,), dtype=float)
        self.q_table_r = np.zeros(self.NUM_BUCKETS + (self.NUM_ACTIONS,), dtype=float)

        self.f_num_steps = 0

        print('NUM_EPISODES: ', self.NUM_EPISODES,
              '\n MAX_STEPS_PER_EPISODE: ', self.MAX_T,
              '\n STREAK_TO_END: ', self.STREAK_TO_END,
              '\n NUM_STEPS_TO_STREAK: ', self.SOLVED_T, '\n')

    def run(self, env):
        self.simulate()
        # self.test_model()

    def simulate(self):

        learning_rate = self.get_learning_rate(0)
        explore_rate = self.get_explore_rate(0)
        discount_factor = 0.99

        num_streaks_c = 0
        num_streaks_r = 0

        self.env.render()

        for episode in range(self.NUM_EPISODES):

            obv_c = self.env.reset_c()
            obv_r = self.env.reset_r()

            state_0_c = self.state_to_bucket(obv_c)
            state_0_r = self.state_to_bucket(obv_r)
            total_reward_c = 0
            total_reward_r = 0

            for t in range(self.MAX_T):
                # CHASER
                self.env.c_turn = True
                action_c = self.select_action(state_0_c, explore_rate)
                obv_c, reward_c, done_c, _ = self.env.step(action_c)
                state_c = self.state_to_bucket(obv_c)
                total_reward_c += reward_c

                best_q = np.amax(self.q_table_c[state_c])
                self.q_table_c[state_0_c + (action_c,)] += learning_rate * (reward_c + discount_factor * (best_q) - self.q_table_c[state_0_c + (action_c,)])
                state_0_c = state_c

                # RUNNER
                self.env.c_turn = False
                action_r = self.select_action(state_0_r, explore_rate)
                obv_r, reward_r, done_r, _ = self.env.step(action_r)
                state_r = self.state_to_bucket(obv_r)
                total_reward_r += reward_r

                best_q = np.amax(self.q_table_r[state_r])
                self.q_table_r[state_0_r + (action_r,)] += learning_rate * (reward_r + discount_factor * (best_q) - self.q_table_r[state_0_r + (action_r,)])
                state_0_r = state_r

                if self.RENDER_MAZE:
                    self.env.render()
                if self.env.is_game_over():
                    sys.exit()
                if done_c:
                    if t <= self.SOLVED_T:
                        num_streaks_c += 1
                        num_streaks_r = 0
                    print("Episode %d finished after %f time steps with total Chaser_reward = %f and "
                          "Runner_reward = %f (Score Chaser - Runner: %d - %d)."
                          % (episode, int(t), total_reward_c, total_reward_r, num_streaks_c, num_streaks_r))
                    break

                elif t >= self.MAX_T - 1:
                    num_streaks_r += 1
                    num_streaks_c = 0
                    print("Episode %d timed out after %f time steps with total Chaser_reward = %f and "
                          "Runner_reward = %f (Score Chaser - Runner: %d - %d)."
                          % (episode, int(t), total_reward_c, total_reward_r, num_streaks_c, num_streaks_r))

            if num_streaks_c > self.STREAK_TO_END:
                self.end_list[0] += 1
                self.f_num_steps = t
                np.save(os.getcwd() + '\\methods\\q_learning\\models\\Q_2Players_' + str(episode) + '_Chaser_' + self.name + '.npy', self.q_table_c)
                num_streaks_c = 0
            if num_streaks_r > self.STREAK_TO_END:
                self.end_list[1] += 1
                self.f_num_steps = t
                num_streaks_r = 0
                np.save(os.getcwd() + '\\methods\\q_learning\\models\\Q_2Players_' + str(episode) + '_Runner_' + self.name + '.npy', self.q_table_r)

            if all(i >= 2 for i in self.end_list):
                break
            if episode == 1 or episode == 1000 or episode == 5000 or episode == 12000 or episode == 19999:
                np.save(os.getcwd() + '\\methods\\q_learning\\models\\Q_2Players_' + str(episode) + '_Chaser_' + self.name + '.npy', self.q_table_c)
                np.save(os.getcwd() + '\\methods\\q_learning\\models\\Q_2Players_' + str(episode) + '_Runner_' + self.name + '.npy', self.q_table_r)

            explore_rate = self.get_explore_rate(episode)
            learning_rate = self.get_learning_rate(episode)

    def test_model(self):
        self.env.render()
        obv_c = self.env.reset_c()
        obv_r = self.env.reset_r()
        state_0_c = self.state_to_bucket(obv_c)
        state_0_r = self.state_to_bucket(obv_r)
        total_reward_c = 0
        total_reward_r = 0
        done_c = False
        done_r = False

        while not done_c:
            self.env.c_turn = True

            action_c = self.select_action(state_0_c, 0)
            obv_c, reward_c, done_c, _ = self.env.step(action_c)
            self.env.c_turn = False
            state_c = self.state_to_bucket(obv_c)
            total_reward_c += reward_c
            state_0_c = state_c

            action_r = self.select_action(state_0_r, 0)
            obv_r, reward_r, done_r, _ = self.env.step(action_r)
            state_r = self.state_to_bucket(obv_r)
            total_reward_r += reward_r
            state_0_r = state_r

            print('Chaser:', action_c, obv_c, done_c)
            print('Runner:', action_r, obv_r, done_r)
            time.sleep(0.5)

            if self.RENDER_MAZE: self.env.render()
            if self.env.is_game_over(): sys.exit()

    def select_action(self, state, explore_rate):
        if random.random() < explore_rate:
            action = self.env.action_space.sample()
        else:
            if self.env.c_turn:
                action = int(np.argmax(self.q_table_c[state]))
            else:
                action = int(np.argmax(self.q_table_r[state]))
        return action

    def get_explore_rate(self, t):
        return max(self.MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/self.DECAY_FACTOR)))

    def get_learning_rate(self, t):
        return max(self.MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/self.DECAY_FACTOR)))

    def state_to_bucket(self, state):
        bucket_indice = []
        for i in range(len(state)):
            if state[i] <= self.STATE_BOUNDS[i][0]:
                bucket_index = 0
            elif state[i] >= self.STATE_BOUNDS[i][1]:
                bucket_index = self.NUM_BUCKETS[i] - 1
            else:
                bound_width = self.STATE_BOUNDS[i][1] - self.STATE_BOUNDS[i][0]
                offset = (self.NUM_BUCKETS[i]-1)*self.STATE_BOUNDS[i][0]/bound_width
                scaling = (self.NUM_BUCKETS[i]-1)/bound_width
                bucket_index = int(round(scaling*state[i] - offset))
            bucket_indice.append(bucket_index)
        return tuple(bucket_indice)

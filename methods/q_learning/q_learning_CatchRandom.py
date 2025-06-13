import sys
import numpy as np
import math
import random
import os
import time


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

        self.q_table = np.zeros(self.NUM_BUCKETS + (self.NUM_ACTIONS,), dtype=float)
        self.f_num_steps = 0

        print('NUM_EPISODES: ', self.NUM_EPISODES,
              '\n MAX_STEPS_PER_EPISODE: ', self.MAX_T,
              '\n STREAK_TO_END: ', self.STREAK_TO_END,
              '\n NUM_STEPS_TO_STREAK: ', self.SOLVED_T, '\n')

    def run(self, env):
        self.simulate()

    def simulate(self):
        learning_rate = self.get_learning_rate(0)
        explore_rate = self.get_explore_rate(0)
        discount_factor = 0.99

        num_streaks = 0

        self.env.render()

        for episode in range(self.NUM_EPISODES):

            obv = self.env.reset()

            state_0 = self.state_to_bucket(obv)
            total_reward = 0

            for t in range(self.MAX_T):

                action = self.select_action(state_0, explore_rate)

                obv, reward, done, _ = self.env.step(action)
                state = self.state_to_bucket(obv)
                total_reward += reward

                best_q = np.amax(self.q_table[state])
                self.q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - self.q_table[state_0 + (action,)])

                state_0 = state

                if self.RENDER_MAZE:
                    self.env.render()

                if self.env.is_game_over():
                    sys.exit()

                if done:
                    print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                          % (episode, t, total_reward, num_streaks))

                    if t <= self.SOLVED_T:
                        num_streaks += 1
                    else:
                        num_streaks = 0
                    break

                elif t >= self.MAX_T - 1:
                    print("Episode %d timed out at %d with total reward = %f."
                          % (episode, t, total_reward))
                    num_streaks = 0

            if num_streaks > self.STREAK_TO_END:
                self.f_num_steps = t
                np.save(os.getcwd() + '\\methods\\q_learning\\models\\Q_CatchRandom_' + self.name + '.npy', self.q_table)
                break

            explore_rate = self.get_explore_rate(episode)
            learning_rate = self.get_learning_rate(episode)

    def test_model(self):
        self.env.render()
        obv = self.env.reset()
        state_0 = self.state_to_bucket(obv)
        total_reward = 0
        done = False
        test_steps = 0

        while not done or test_steps < self.MAX_T:
            action = self.select_action(state_0, 0)
            obv, reward, done, _ = self.env.step(action)
            print(action, obv, done)
            time.sleep(0.5)
            state = self.state_to_bucket(obv)
            total_reward += reward
            state_0 = state
            test_steps += 1

            if self.RENDER_MAZE: self.env.render()
            if self.env.is_game_over(): sys.exit()

    def select_action(self, state, explore_rate):
        if random.random() < explore_rate:
            action = self.env.action_space.sample()
        else:
            action = int(np.argmax(self.q_table[state]))
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

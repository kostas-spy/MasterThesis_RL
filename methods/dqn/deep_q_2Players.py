import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding, Reshape
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint, FileLogger, TrainEpisodeLogger, TrainIntervalLogger


class DeepQNetwork:
    def __init__(self, name, env):
        self.dqn = None
        self.env = env
        self.name = name
        self.problem = env.problem
        self.state_size = env.observation_space
        self.action_size = env.action_space.n
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.2  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.max_reward = -100
        self.save_path_c = os.getcwd() + '\\methods\\dqn\\models\\weights\\DQN_2Players_Chaser_' + name + '.h5f'
        self.save_path_r = os.getcwd() + '\\methods\\dqn\\models\\weights\\DQN_2Players_Runner_' + name + '.h5f'

        # Model Params
        self.seq_memory = 100000
        self.nb_steps_c = 200000
        self.nb_steps_r = 250000
        self.log_interval = 1500
        self.max_steps = 30
        self.warm_steps = 5000
        self.learning_rate = 0.0001
        self.model = self.model_v1()

    def model_v0(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.state_size.shape))
        model.add(Dense(20, activation='selu'))
        model.add(Dense(20, activation='selu'))
        model.add(Dense(self.action_size, activation='softmax'))
        print(model.summary())
        return model

    def model_v1(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.state_size.shape))
        model.add(Dense(50, activation='selu'))
        model.add(Dense(50, activation='selu'))
        model.add(Dense(50, activation='selu'))
        model.add(Dense(20, activation='selu'))
        model.add(Dense(self.action_size, activation='softmax'))
        print(model.summary())
        return model

    def run(self):
        # Chaser
        self.env.c_turn = True
        memory = SequentialMemory(limit=self.seq_memory, window_length=1)
        policy = EpsGreedyQPolicy(eps=self.epsilon)
        self.dqn = DQNAgent(model=self.model, nb_actions=self.action_size, memory=memory, nb_steps_warmup=self.warm_steps, target_model_update=1e-2, policy=policy, gamma=self.gamma)
        self.dqn.compile(Adam(lr=self.learning_rate), metrics=['mse'])
        history_c = self.dqn.fit(self.env, nb_steps=self.nb_steps_c, visualize=True, verbose=1, nb_max_episode_steps=self.max_steps, log_interval=self.log_interval, callbacks=[TrainIntervalLogger(model_name='2Players_Chaser')])

        self.dqn.test(self.env, nb_episodes=5, visualize=True, nb_max_episode_steps=self.max_steps)
        self.dqn.save_weights(self.save_path_c, overwrite=True)

        plt.plot(history_c.history['episode_reward'])
        plt.title('model reward')
        plt.ylabel('reward')
        plt.xlabel('epoch')
        plt.plot()
        plt.savefig(os.getcwd() + '\\methods\\dqn\\models\\Chaser_reward_obst_2.png')

        # Runner
        self.env.c_turn = False
        memory = SequentialMemory(limit=self.seq_memory, window_length=1)
        policy = EpsGreedyQPolicy(eps=self.epsilon)
        self.dqn = DQNAgent(model=self.model, nb_actions=self.action_size, memory=memory, nb_steps_warmup=self.warm_steps, target_model_update=1e-2, policy=policy, gamma=self.gamma)
        self.dqn.compile(Adam(lr=self.learning_rate), metrics=['mse'])
        history_r = self.dqn.fit(self.env, nb_steps=self.nb_steps_r, visualize=True, verbose=1, nb_max_episode_steps=self.max_steps, log_interval=self.log_interval, callbacks=[TrainIntervalLogger(model_name='2Players_Runner')])

        self.dqn.test(self.env, nb_episodes=5, visualize=True, nb_max_episode_steps=self.max_steps)
        self.dqn.save_weights(self.save_path_r, overwrite=True)

        plt.plot(history_r.history['episode_reward'])
        plt.title('model reward')
        plt.ylabel('reward')
        plt.xlabel('epoch')
        plt.plot()
        plt.savefig(os.getcwd() + '\\methods\\dqn\\models\\Runner_reward_obst_2.png')

    def test(self):
        memory = SequentialMemory(limit=self.seq_memory, window_length=1)
        policy = EpsGreedyQPolicy(eps=self.epsilon)

        test_model_c = DQNAgent(model=self.model, nb_actions=self.action_size, memory=memory, nb_steps_warmup=self.warm_steps, target_model_update=1e-2, policy=policy, gamma=self.gamma)
        test_model_c.compile(Adam(lr=self.learning_rate), metrics=['mse'])
        test_model_c.load_weights(filepath=self.save_path_c)
        test_model_c.test(self.env, nb_episodes=10, visualize=True, nb_max_episode_steps=self.max_steps)

        test_model_r = DQNAgent(model=self.model, nb_actions=self.action_size, memory=memory, nb_steps_warmup=self.warm_steps, target_model_update=1e-2, policy=policy, gamma=self.gamma)
        test_model_r.compile(Adam(lr=self.learning_rate), metrics=['mse'])
        test_model_r.load_weights(filepath=self.save_path_r)
        test_model_r.test(self.env, nb_episodes=10, visualize=True, nb_max_episode_steps=self.max_steps)


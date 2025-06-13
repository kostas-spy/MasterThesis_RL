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
        self.save_path = os.getcwd() + '\\methods\\dqn\\models\\weights\\DQN_CatchRandom_' + name + '.h5f'

        # Model Params
        self.seq_memory = 100000
        self.nb_steps = 40000
        self.log_interval = 1000
        self.max_steps = 100
        self.warm_steps = 10000
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

    def build_callbacks(self):
        checkpoint_weights_filename = os.getcwd() + '\\methods\\dqn\\models\\chks\\chkDQN_CatchRandom_' + self.name + '_{step}.h5f'
        log_filename = os.getcwd() + '\\methods\\dqn\\chks\\weights\\LOG_DQN_CatchRandom_{}_log.json'.format(self.name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=1000)]
        tel = TrainIntervalLogger(model_name='CatchRandom')
        callbacks += [tel]

        return callbacks

    def run(self):
        memory = SequentialMemory(limit=self.seq_memory, window_length=1)
        policy = EpsGreedyQPolicy(eps=self.epsilon)
        self.dqn = DQNAgent(model=self.model, nb_actions=self.action_size, memory=memory, nb_steps_warmup=self.warm_steps, target_model_update=1e-2, policy=policy, gamma=self.gamma)
        self.dqn.compile(Adam(lr=self.learning_rate), metrics=['mse'])

        history = self.dqn.fit(self.env, nb_steps=self.nb_steps, visualize=True, verbose=1, nb_max_episode_steps=self.max_steps, log_interval=self.log_interval, callbacks=[TrainIntervalLogger(model_name='CatchRandom')])

        plt.plot(history.history['episode_reward'])
        plt.title('model reward')
        plt.ylabel('reward')
        plt.xlabel('epoch')
        plt.show()

        self.dqn.test(self.env, nb_episodes=5, visualize=True, nb_max_episode_steps=self.max_steps)
        self.dqn.save_weights(self.save_path, overwrite=True)

    def test(self):
        memory = SequentialMemory(limit=self.seq_memory, window_length=1)
        policy = EpsGreedyQPolicy(eps=self.epsilon)
        test_model = DQNAgent(model=self.model, nb_actions=self.action_size, memory=memory, nb_steps_warmup=self.warm_steps, target_model_update=1e-2, policy=policy, gamma=self.gamma)
        test_model.compile(Adam(lr=self.learning_rate), metrics=['mae'])
        test_model.load_weights(filepath=self.save_path)
        test_model.test(self.env, nb_episodes=10, visualize=True, nb_max_episode_steps=self.max_steps)


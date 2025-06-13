import os
import importlib
import time
import config as c
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
start_time = time.time()


if __name__ == "__main__":
    problem_to_solve = c.PROBLEM
    name = c.MAP_NAME
    train_mode = c.TRAIN_MODE

    # Environment Paths
    map_path = os.getcwd() + '\\methods\\envs\\map_samples\\' + name + '\\'
    map_file_walls = map_path + name + '_walls.npy'
    map_file_obstacles = map_path + name + '_obst.npy'
    env_import = importlib.import_module('methods.envs.map_env_' + problem_to_solve)
    env_ = env_import.MapEnv
    env_view2d_import = importlib.import_module('methods.envs.map_view_2d_' + problem_to_solve)
    env_view2d_ = env_view2d_import.MapView2D

    if train_mode == 'Q':
        Q_module = importlib.import_module('methods.q_learning.q_learning_' + problem_to_solve)
        QLearning = Q_module.QLearning
        env = env_(map_name=name, map_file_wall=map_file_walls, map_file_obst=map_file_obstacles,
                   enable_render=True, problem=problem_to_solve)
        q = QLearning(env, name=name)
        q.run(env)
    elif train_mode == 'DQN':
        DQN_module = importlib.import_module('methods.dqn.deep_q_' + problem_to_solve)
        DeepQNetwork = DQN_module.DeepQNetwork
        env = env_(map_name=name, map_file_wall=map_file_walls, map_file_obst=map_file_obstacles,
                   enable_render=True, problem=problem_to_solve)
        dqn = DeepQNetwork(name=name, env=env)
        dqn.run()
    elif train_mode == 'NEAT':
        NEAT_module = importlib.import_module('methods.neat_learning.feedforward_' + problem_to_solve)
        NEATEvolve = NEAT_module.NEATEvolve
        env = env_(map_name=name, map_file_wall=map_file_walls, map_file_obst=map_file_obstacles,
                   enable_render=False, problem=problem_to_solve)
        neat_ = NEATEvolve(name=name, env=env)
        neat_.run()
    else:
        print('Please provide one of the following models to train (Q - DQN - NEAT)')

    map = env_view2d_(map_name=name, map_file_path_wall=map_file_walls, map_file_path_obst=map_file_obstacles,
                      screen_size=c.SCREEN_SIZE, enable_render=c.ENABLE_RENDER)
    map.update()
    print("--- %s seconds ---" % (time.time() - start_time))
    input("Enter any key to quit.")

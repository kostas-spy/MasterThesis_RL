# Thesis - Simulation and Comparison of Reinforcement Learning Algorithms

On this project, we focus on comparing three main Reinforcement Learning algorithms Q, Deep Q and NEAT.

## How to Run
* Create new map from map_generator.py or load it in main.py or load an existing one.
* Run main.py with the correct problem_to_solve, map_name, and training method (Q / NEAT).


## Code Description
* **main**
   * Run main program.
   * Choose Q, DeepQ or NEAT algorithms 
   * Select map file to run from
* **main_test**
   * Test pretrained model from main program.
* **config**
   * Configuration setting (map, algorithm, size, settings)
* **map_env** 
   * Build main gym.env functions 
   * Calls map_view_2d class
* **map_view_2d**
   * Designs map with pygame
   * Calls map_build class
* **map_build**
   * Saves and loads map
   * Checks walls/obstacles
   * Generate map (disabled)
    

### Map file must be folder in map_samples
* *_obst.npy
* *_walls.npy


##PROBLEMS to solve:
* **CatchStatic** : agent reaches goal (find shortest path)
* **CatchRandom** : agent reaches moving goal
* **2Players** : Final Version of the game, trains both agents


## Author / Support

RL_map is a Reinforcement Learning Implementation created by Kostas Spyropoulos.

Questions can be directed to kostas.spy93@gmail.com

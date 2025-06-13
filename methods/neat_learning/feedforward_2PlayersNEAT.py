import os
import neat
import multiprocessing
import pickle
from methods.envs.visualize_plots import plot_stats, plot_species, draw_net
import time
import numpy as np
import config as c


class NEATEvolve:

    COMPASS = {"N": (-1, 0), "S": (0, 1), "W": (0, -1), "E": (1, 0)}

    def __init__(self, name=None, env=None):
        self.env = env
        self.name = name
        self.runs_per_net = 20
        self.num_generations = 4000
        self.t = 0.0
        self.size = self.env.map_size[0]
        self.c_turn = True

        # Load the config file
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'models/2Players-config-feedforward')
        self.winner_path_c = os.path.join(local_dir, 'models/NEAT_2Players_Chaser_' + self.name)
        self.winner_path_r = os.path.join(local_dir, 'models/NEAT_2Players_Runner_' + self.name)
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    def run(self):
        # Chaser
        pop = neat.Population(self.config)
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.StdOutReporter(True))

        if c.MULTI_PROCESS:
            pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), self.eval_genome)
            winner = pop.run(pe.evaluate, self.num_generations)
        else:
            winner = pop.run(self.eval_genomes, self.num_generations)

        print(winner)
        self.plot_creator(stats, winner, 'Chaser')
        with open(self.winner_path_c, 'wb') as f:
            pickle.dump(winner, f)

        self.c_turn = False

        # Runner
        self.t = 0.0
        pop = neat.Population(self.config)
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.StdOutReporter(True))

        if c.MULTI_PROCESS:
            pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), self.eval_genome)
            winner = pop.run(pe.evaluate, self.num_generations)
        else:
            winner = pop.run(self.eval_genomes, self.num_generations)

        print(winner)
        self.plot_creator(stats, winner, 'Runner')
        with open(self.winner_path_r, 'wb') as f:
            pickle.dump(winner, f)

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = self.eval_genome(genome, config)

    # Fitness Creator
    # Use the NN network phenotype and the discrete actuator force function.
    def eval_genome(self, genome, config):

        if self.c_turn:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            # net = neat.nn.RecurrentNetwork.create(genome, config)
            fitnesses = []

            for runs in range(self.runs_per_net):
                observation = self.env.reset_c()
                fitness = 0.0
                done = False
                self.t = 0.0
                while not done and self.t < 25:
                    action = int(np.argmax(net.activate(observation)))
                    observation, reward, done, info = self.env.step(action)
                    # print(int(self.t), ': Observation:', observation, '- Reward:', reward, '- Action:', action, '/', list(self.COMPASS)[action], '- Fitness:', fitness)
                    # input('*+*+*+*+*+*+*+*+*+*+' * 5)

                    fitness += reward
                    self.t += 1
                fitnesses.append(fitness)
            return np.mean(fitnesses)
        else:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            # net = neat.nn.RecurrentNetwork.create(genome, config)
            fitnesses = []

            for runs in range(self.runs_per_net):
                observation = self.env.reset_r()
                fitness = 0.0
                done = False
                self.t = 0.0
                while not done and self.t < 25:
                    action = int(np.argmax(net.activate(observation)))
                    observation, reward, done, info = self.env.step(action)
                    # print(int(self.t), ': Observation:', observation, '- Reward:', reward, '- Action:', action, '/', list(self.COMPASS)[action], '- Fitness:', fitness)
                    # input('*+*+*+*+*+*+*+*+*+*+' * 5)

                    fitness += reward
                    self.t += 1
                fitnesses.append(fitness)
            return np.mean(fitnesses)

    # Test Winner model
    def test(self):
        with open(self.winner_path_c, 'rb') as f:
            model_c = pickle.load(f)
        with open(self.winner_path_r, 'rb') as f:
            model_r = pickle.load(f)

        net_c = neat.nn.FeedForwardNetwork.create(model_c, self.config)
        net_r = neat.nn.FeedForwardNetwork.create(model_r, self.config)
        observation_c = self.env.reset_c()
        observation_r = self.env.reset_r()
        print()
        done = False
        while not done:
            action_c = int(np.argmax(net_c.activate(observation_c)))
            observation_c, reward_c, done_c, info = self.env.step(action_c)

            action_r = int(np.argmax(net_r.activate(observation_r)))
            observation_r, reward_r, done_r, info = self.env.step(action_r)

            print(action_c, action_r)
            print(observation_c, observation_r)
            print('------'*10)
            self.env.render()
            time.sleep(0.5)

    def plot_creator(self, stats, winner, agent_name):
        # Visualize Plots and Nets
        node_names = {-1: 'x_agent', -2: 'y_agent', -3: 'x_enemy', -4: 'y_enemy', 0: 'N', 1: 'S', 2: 'W', 3: 'E'}
        save_path = os.path.join(os.path.dirname(__file__), 'models\\')
        plot_stats(stats, ylog=True, view=True, filename=save_path + "\\2Players_" + agent_name + "-feedforward-fitness.png")
        plot_species(stats, view=True, filename=save_path + "\\2Players_" + agent_name + "-feedforward-speciation.png")

        draw_net(self.config, winner, view=True, node_names=node_names, filename=save_path + "\\2Players_" + agent_name + "-winner-feedforward.gv")

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
        self.runs_per_net = 30
        self.num_generations = 15000
        self.t = 0.0
        self.size = self.env.map_size[0]
        self.one_hot_enc = np.zeros(self.size ** 4)

        # Load the config file
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'models/CatchRandom-config-feedforward')
        self.winner_path = os.path.join(local_dir, 'models/NEAT_CatchRandom_' + self.name)
        print(self.winner_path)
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_path)

    # Main Run Method
    def run(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'models/CatchRandom-config-feedforward')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_path)

        # Create the population
        pop = neat.Population(self.config)

        # Report
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.StdOutReporter(True))

        if c.MULTI_PROCESS:
            # Multiprocessing
            pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), self.eval_genome)    # (1, self.eval_genome)
            winner = pop.run(pe.evaluate, self.num_generations)
        else:
            # Single Processing
            winner = pop.run(self.eval_genomes, self.num_generations)

        print(winner)
        self.plot_creator(stats, winner)

        # Save the CatchStatic_winner.
        with open(self.winner_path, 'wb') as f:
            pickle.dump(winner, f)

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = self.eval_genome(genome, config)

    # Fitness Creator
    # Use the NN network phenotype and the discrete actuator force function.
    def eval_genome(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # net = neat.nn.RecurrentNetwork.create(genome, config)
        fitnesses = []

        for runs in range(self.runs_per_net):
            observation = self.env.reset()
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
        with open(self.winner_path, 'rb') as f:
            model = pickle.load(f)

        print('Loaded Genome:', self.winner_path)
        print(c)
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'models/CatchRandom-config-feedforward')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_path)

        net = neat.nn.FeedForwardNetwork.create(model, self.config)
        observation = self.env.reset()

        done = False
        while not done:
            action = int(np.argmax(net.activate(observation)))
            observation, reward, done, info = self.env.step(action)
            print(action)
            print(observation)
            print('------'*10)
            self.env.render()
            time.sleep(0.5)

    def plot_creator(self, stats, winner):
        # Visualize Plots and Nets
        node_names = {-1: 'x_agent', -2: 'y_agent', -3: 'x_enemy', -4: 'y_enemy', 0: 'N', 1: 'S', 2: 'W', 3: 'E'}
        save_path = os.path.join(os.path.dirname(__file__), 'models\\')
        plot_stats(stats, ylog=True, view=True, filename=save_path + "\\feedforward-fitness.svg")
        plot_species(stats, view=True, filename=save_path + "\\feedforward-speciation.svg")

        draw_net(self.config, winner, True, node_names=node_names)
        draw_net(self.config, winner, view=True, node_names=node_names, filename=save_path + "\\winner-feedforward.gv")
        draw_net(self.config, winner, view=True, node_names=node_names, filename=save_path + "\\winner-feedforward-enabled.gv", show_disabled=False)

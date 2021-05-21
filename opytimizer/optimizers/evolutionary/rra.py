"""Runner-Root Algorithm.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.distribution as dist
import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class RRA(Optimizer):
    """An RRA class, inherited from Optimizer.

    This is the designed class to define RRA-related
    variables and methods.

    References:
        F. Merrikh-Bayat.
        The runner-root algorithm: A metaheuristic for solving unimodal and
        multimodal optimization problems inspired by runners and roots of plants in nature.
        Applied Soft Computing (2015).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> RRA.')

        # Overrides its parent class with the receiving params
        super(RRA, self).__init__()

        #
        self.d_runner = 2

        #
        self.d_root = 0.01

        #
        self.max_stall = 1000

        #
        self.tol = 0.01

        self.last_best_fit = c.FLOAT_MAX

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    def _roulette_selection(self, fitness):
        """Performs a roulette selection on the population (p. 8).

        Args:
            n_agents (int): Number of agents allowed in the space.
            fitness (list): A fitness list of every agent.

        Returns:
            The selected indexes of the population.

        """


        # Defines the maximum fitness of current generation
        max_fitness = np.max(fitness)

        # Re-arrange the list of fitness by inverting it
        # Note that we apply a trick due to it being designed for minimization
        # f'(x) = f_max - f(x)
        inv_fitness = [max_fitness - fit + c.EPSILON for fit in fitness]

        # Calculates the total inverted fitness
        total_fitness = np.sum(inv_fitness)

        # Calculates the probability of each inverted fitness
        probs = [fit / total_fitness for fit in inv_fitness]

        # Performs the selection process
        selected = dist.generate_choice_distribution(len(probs), probs, 1)

        return selected

    def update(self, space, function):
        space.agents.sort(key=lambda x: x.fit)
        self.last_best_fit = space.agents[0].fit

        daughters = copy.deepcopy(space.agents)

        for i, daughter in enumerate(daughters):
            if i != 0:
                r1 = r.generate_uniform_random_number()
                daughter.position += self.d_runner * r1
                daughter.clip_by_bound()
                daughter.fit = function(daughter.position)

        daughters.sort(key=lambda x: x.fit)

        if np.fabs((self.last_best_fit - daughters[0].fit) / self.last_best_fit) < self.tol:
            stall_occurence = True
        else:
            stall_occurence = False
            self.stall_counter = 0

        if stall_occurence:
            for j in range(daughters[0].n_variables):
                d = copy.deepcopy(daughters[0])
                r1 = r.generate_uniform_random_number()
                d.position[j] += self.d_runner * r1
                d.clip_by_bound()
                d.fit = function(d.position)

                if d.fit < daughters[0].fit:
                    daughters[0].position = copy.deepcopy(d.position)
                    daughters[0].fit = copy.deepcopy(d.fit)

            for j in range(daughters[0].n_variables):
                d = copy.deepcopy(daughters[0])
                r1 = r.generate_uniform_random_number()
                d.position[j] += self.d_root * (r1 - 0.5)
                d.clip_by_bound()
                d.fit = function(d.position)

                if d.fit < daughters[0].fit:
                    daughters[0].position = copy.deepcopy(d.position)
                    daughters[0].fit = copy.deepcopy(d.fit)

        fit = [daughter.fit for daughter in daughters]

        

        space.agents[0] = copy.deepcopy(daughters[0])

        for i, agent in enumerate(space.agents):
            if i != 0:
                idx = self._roulette_selection(fit)[0]
                space.agents[i] = copy.deepcopy(daughters[idx])

        if np.fabs((self.last_best_fit - daughters[0].fit) / self.last_best_fit) < self.tol:
            self.stall_counter += 1
        else:
            self.stall_counter = 0
            
        if self.stall_counter == self.max_stall:
            for agent in space.agents:
                agent.fill_with_uniform()
                # agent.fit = function(agent.position)
            self.stall_counter = 0
        print(self.last_best_fit)

        # print(self.stall_counter)
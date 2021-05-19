"""Parasitism-Predation Algorithm.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.distribution as d
import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class PPA(Optimizer):
    """A PPA class, inherited from Optimizer.

    This is the designed class to define PPA-related
    variables and methods.

    References:
        A. Mohamed et al. Parasitism – Predation algorithm (PPA): A novel approach for feature selection.
        Ain Shams Engineering Journal (2020).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> PPA.')

        # Overrides its parent class with the receiving params
        super(PPA, self).__init__()

        #
        self.r1 = 1

        #
        self.r2 = 0.1

        #
        self.r3 = 0.1

        #
        self.alpha1 = 0.2

        #
        self.alpha2 = 0.25

        #
        self.beta1 = 0.1

        #
        self.beta2 = 0.1

        #
        self.c1 = 0.1

        #
        self.c2 = 0.1

        #
        self.d1 = 0.01

        #
        self.d2 = 0.01

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    def compile(self, space):
        self.velocity = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

    def _growth_rate(self, n_agents, iteration, n_iterations):
        """
        """

        #
        n_crows = np.round(n_agents * (2/3 - iteration * ((2/3 - 1/2) / n_iterations)))

        #
        n_cats = np.round(n_agents * (0.01 + iteration * ((1/3 - 0.01) / n_iterations)))

        #
        n_cuckoos = n_agents - n_crows - n_cats

        return int(n_crows), int(n_cats), int(n_cuckoos)

    def update(self, space, iteration, n_iterations):
        #
        n_crows, n_cats, n_cuckoos = self._growth_rate(space.n_agents, iteration, n_iterations)

        # print(n_crows, n_cats, n_cuckoos)

        #
        for i, agent in enumerate(space.agents[:n_crows]):
            #
            idx = r.generate_integer_random_number(high=space.n_agents, exclude_value=i)

            # (eq. 7)
            step = d.generate_levy_distribution(size=agent.n_variables)
            step = np.expand_dims(step, axis=1)

            # (eq. 6 and 8)
            agent.position = 0.01 * step * (space.agents[idx].position - agent.position)

            #
            agent.clip_by_bound()

        #
        p = iteration / n_iterations

        # Calculates a list of current trees' fitness
        fitness = [agent.fit for agent in space.agents[n_crows:n_crows+n_cuckoos]]

        #
        for agent in space.agents[n_crows:n_crows+n_cuckoos]:
            #
            s = g.tournament_selection(fitness, 1)[0]

            #
            i = r.generate_integer_random_number(0, space.n_agents)
            j = r.generate_integer_random_number(0, space.n_agents, exclude_value=i)

            # (eq. 12)
            k = r.generate_uniform_random_number(size=agent.n_variables) > p
            k = np.expand_dims(k, -1)

            # (eq. 11)
            rand = r.generate_uniform_random_number()
            S_g = (space.agents[i].position - space.agents[j].position) * rand

            # (eq. 10)
            agent.position = space.agents[s].position + S_g * k

            agent.clip_by_bound()
            

        #
        _c = 2 - iteration / n_iterations

        #
        for i, agent in enumerate(space.agents[n_crows+n_cuckoos:]):
            idx = i + n_crows + n_cuckoos
            r1 = r.generate_uniform_random_number()
            self.velocity[idx] += r1 * _c * (space.best_agent.position - agent.position)

            agent.position += self.velocity[idx]

            agent.clip_by_bound()


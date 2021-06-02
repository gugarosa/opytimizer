"""Artificial Flora.
"""

import copy

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core import Optimizer

logger = l.get_logger(__name__)


class AF(Optimizer):
    """An AF class, inherited from Optimizer.

    This is the designed class to define AF-related
    variables and methods.

    References:
        L. Cheng, W. Xue-han and Y. Wang. Artificial flora (AF) optimization algorithm.
        Applied Sciences (2018).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> AF.')

        # Overrides its parent class with the receiving params
        super(AF, self).__init__()

        # First learning coefficient
        self.c1 = 0.75

        # Second learning coefficient
        self.c2 = 1.25

        # Amount of branches
        self.m = 10

        # Selective probability
        self.Q = 0.75

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    @property
    def c1(self):
        """float: First learning coefficient.

        """

        return self._c1

    @c1.setter
    def c1(self, c1):
        if not isinstance(c1, (float, int)):
            raise e.TypeError('`c1` should be a float or integer')
        if c1 < 0:
            raise e.ValueError('`c1` should be >= 0')

        self._c1 = c1

    @property
    def c2(self):
        """float: Second learning coefficient.

        """

        return self._c2

    @c2.setter
    def c2(self, c2):
        if not isinstance(c2, (float, int)):
            raise e.TypeError('`c2` should be a float or integer')
        if c2 < 0:
            raise e.ValueError('`c2` should be >= 0')

        self._c2 = c2

    @property
    def m(self):
        """int: Amount of branches.

        """

        return self._m

    @m.setter
    def m(self, m):
        if not isinstance(m, int):
            raise e.TypeError('`m` should be an integer')
        if m <= 0:
            raise e.ValueError('`m` should be > 0')

        self._m = m

    @property
    def Q(self):
        """float: Selective probability.

        """

        return self._Q

    @Q.setter
    def Q(self, Q):
        if not isinstance(Q, (float, int)):
            raise e.TypeError('`Q` should be a float or integer')
        if Q < 0 or Q > 1:
            raise e.ValueError('`Q` should be between 0 and 1')

        self._Q = Q

    def compile(self, space):
        """Compiles additional information that is used by this optimizer.

        Args:
            space (Space): A Space object containing meta-information.

        """

        # Array of parent distances
        self.p_distance = r.generate_uniform_random_number(size=space.n_agents)

        # Array of grand-parent distances
        self.g_distance = r.generate_uniform_random_number(size=space.n_agents)

    def update(self, space, function):
        """Wraps Artificial Flora over all agents and variables.

        Args:
            space (Space): Space containing agents and update-related information.
            function (Function): A Function object that will be used as the objective function.

        """
        
        # Sorts the agents
        space.agents.sort(key=lambda x: x.fit)

        # Creates a list of new agents
        new_agents = []

        # Iterates thorugh all agents
        for i, agent in enumerate(space.agents):
            # Iterates through amount of branches
            for _ in range(self.m):
                # Makes a copy of current agent
                a = copy.deepcopy(agent)

                # Generates random numbers
                r1 = r.generate_uniform_random_number()
                r2 = r.generate_uniform_random_number()

                #
                distance = self.g_distance[i] * r1 * \
                    self.c1 + self.p_distance[i] * r2 * self.c2

                # print(distance)

                #
                D = r.generate_gaussian_random_number(variance=distance)

                a.position += D
                a.clip_by_bound()

                #
                a.fit = function(a.position)

                # print(a.fit)

                # print(D)

                #
                p = np.fabs(np.sqrt(a.fit / space.agents[-1].fit)) * self.Q

                #
                r3 = r.generate_uniform_random_number()

                if r3 < p:
                    new_agents.append(a)

            #
            self.g_distance[i] = self.p_distance[i]
            self.p_distance[i] = np.sqrt(
                np.sum((agent.position - a.position) ** 2) / agent.n_variables)

            print(self.g_distance[i], self.p_distance[i])

        space.agents += new_agents
        space.agents.sort(key=lambda x: x.fit)
        space.agents = space.agents[:space.n_agents]

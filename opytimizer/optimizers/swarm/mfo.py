"""Moth-Flame Optimization.
"""

import copy

import numpy as np

import opytimizer.math.random as rnd
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class MFO(Optimizer):
    """A MFO class, inherited from Optimizer.

    This is the designed class to define MFO-related
    variables and methods.

    References:
        S. Mirjalili. Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm.
        Knowledge-Based Systems (2015).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> MFO.')

        # Overrides its parent class with the receiving params
        super(MFO, self).__init__()

        # Spiral constant
        self.b = 1

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    @property
    def b(self):
        """float: Spiral constant.

        """

        return self._b

    @b.setter
    def b(self, b):
        if not isinstance(b, (float, int)):
            raise e.TypeError('`b` should be a float or integer')
        if b < 0:
            raise e.ValueError('`b` should be >= 0')

        self._b = b

    def update(self, space, iteration, n_iterations):
        """Wraps Moth-Flame Optimization over all agents and variables.

        Args:
            space (Space): Space containing agents and update-related information.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Makes a deepcopy of current population
        flames = copy.deepcopy(space.agents)

        # Sorts the flames
        flames.sort(key=lambda x: x.fit)

        # Calculates the number of flames (eq. 3.14)
        n_flames = int(len(flames) - iteration * ((len(flames) - 1) / n_iterations)) - 1

        # Calculates the convergence constant
        r = -1 + iteration * (-1 / n_iterations)

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Iterates through every decision variable
            for j in range(agent.n_variables):
                # Generates a random `t`
                t = rnd.generate_uniform_random_number(r, 1)

                # Checks if current moth should be updated with corresponding flame
                if i < n_flames:
                    # Calculates the distance (eq. 3.13)
                    D = np.fabs(flames[i].position[j] - agent.position[j])

                    # Updates current agent's position (eq. 3.12)
                    agent.position[j] = D * np.exp(self.b * t) * \
                        np.cos(2 * np.pi * t) + flames[i].position[j]

                # If current moth should be updated with best flame
                else:
                    # Calculates the distance (eq. 3.13)
                    D = np.fabs(flames[0].position[j] - agent.position[j])

                    # Updates current agent's position (eq. 3.12)
                    agent.position[j] = D * np.exp(self.b * t) * \
                        np.cos(2 * np.pi * t) + flames[0].position[j]

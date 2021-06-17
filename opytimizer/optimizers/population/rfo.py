"""Red Fox Optimization.
"""

import copy

import numpy as np

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core import Optimizer

logger = l.get_logger(__name__)


class RFO(Optimizer):
    """A RFO class, inherited from Optimizer.

    This is the designed class to define RFO-related
    variables and methods.

    References:
        D. Polap and M. Woźniak. Red fox optimization algorithm.
        Expert Systems with Applications (2021).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> RFO.')

        # Overrides its parent class with the receiving params
        super(RFO, self).__init__()

        # Observation angle
        self.phi = r.generate_uniform_random_number(0, 2*np.pi)[0]

        # Weather condition
        self.theta = r.generate_uniform_random_number()[0]

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    @property
    def phi(self):
        """float: Observation angle.

        """

        return self._phi

    @phi.setter
    def phi(self, phi):
        if not isinstance(phi, (float, int)):
            raise e.TypeError('`phi` should be a float or integer')
        if phi < 0 or phi > 2*np.pi:
            raise e.ValueError('`phi` should be between 0 and 2PI')

        self._phi = phi

    @property
    def theta(self):
        """float: Weather condition.

        """

        return self._theta

    @theta.setter
    def theta(self, theta):
        if not isinstance(theta, (float, int)):
            raise e.TypeError('`theta` should be a float or integer')
        if theta < 0 or theta > 1:
            raise e.ValueError('`theta` should be between 0 and 1')

        self._theta = theta

    def _rellocation(self, agent, best_agent, function):
        """Performs the fox rellocation procedure.

        Args:
            agent (Agent): Current agent.
            best_agent (Agent): Best agent.
            function (Function): A Function object that will be used as the objective function.

        """

        # Creates a temporary agent
        temp = copy.deepcopy(agent)

        # Calculates the square root of euclidean distance between agent and best agent (eq. 1)
        distance = np.sqrt(g.euclidean_distance(
            temp.position, best_agent.position))

        # Randomly selects the scaling hyperparameter
        alpha = r.generate_uniform_random_number(0, distance)

        # Calculates individual reallocation (eq. 2)
        temp.position += alpha * np.sign(best_agent.position - temp.position)

        # Checks agent's limits
        temp.clip_by_bound()

        # Calculates the fitness for the temporary position
        temp.fit = function(temp.position)

        # If new fitness is better than agent's fitness
        if temp.fit < agent.fit:
            # Copies its position and fitness to the agent
            agent.position = copy.deepcopy(temp.position)
            agent.fit = copy.deepcopy(temp.fit)

    def _noticing(self, agent, best_agent, function, alpha):
        """Performs the fox noticing procedure.

        Args:
            agent (Agent): Current agent.
            best_agent (Agent): Best agent.
            function (Function): A Function object that will be used as the objective function.
            alpha (float): Scaling parameter.

        """

        # Defines the noticing parameter
        mu = r.generate_uniform_random_number()

        # If noticing is higher than 0.75
        if mu > 0.75:
            # If observation angle is different than zero
            if self.phi != 0:
                # Calculates fox observation radius (eq. 4 - top)
                radius = alpha * np.sin(self.phi) / self.phi

            # If observation angle equals to zero
            else:
                # Calculates fox observation radius (eq. 4 - bottom)
                radius = self.theta

            phi = r.generate_uniform_random_number(
                0, 2 * np.pi, agent.n_variables)

            # Calculate reallocation according to Eq. (5)
            for j in range(agent.n_variables):

                if j == 0:
                    agent.position[j] = alpha * radius * \
                        np.cos(phi[j]) + agent.position[j]
                else:

                    summation = 0

                    for i in range(j):
                        summation += np.sin(phi[i])

                    agent.position[j] = alpha * radius * summation + \
                        alpha * radius * np.cos(phi[j]) + agent.position[j]

    def update(self, space, function):
        """Wraps Red Fox Optimization over all agents and variables.

        Args:
            space (Space): Space containing agents and update-related information.
            function (Function): A Function object that will be used as the objective function.

        """

        # Defines the scaling parameter
        alpha = r.generate_uniform_random_number(0, 0.2)

        # Iterates through all agents
        for agent in space.agents:
            # Performs the fox rellocation procedure
            self._rellocation(agent, space.best_agent, function)

            # Performs the fox noticing procedure
            # self._noticing(agent, space.best_agent, function, alpha)

        # Sorts agents
        space.agents.sort(key=lambda x: x.fit)

        # Calculating the center of the habitat according to Eq. (6)
        habitat_center = (
            space.agents[0].position + space.agents[1].position) / 2

        habitat_diameter = np.sqrt(g.euclidean_distance(
            space.agents[0].position, space.agents[1].position))

        # Generates random number
        k = r.generate_uniform_random_number()

        for i in range(-int(space.n_agents * 0.05), 0, 1):

            if k >= 0.45:

                # New nomadic individual
                space.agents[i].fill_with_uniform()
                space.agents[i].position += habitat_center + \
                    habitat_diameter / 2

            else:

                # Reproduction of the alpha couple
                space.agents[i].position = k * \
                    (space.agents[0].position + space.agents[1].position) / 2

            # Checks agent's limits
            space.agents[i].clip_by_bound()

            # Calculates the fitness
            space.agents[i].fit = function(space.agents[i].position)

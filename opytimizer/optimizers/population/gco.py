"""Germinal Center Optimization.
"""

import copy

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class GCO(Optimizer):
    """A GCO class, inherited from Optimizer.

    This is the designed class to define GCO-related
    variables and methods.

    References:
        C. Villaseñor et al. Germinal center optimization algorithm.
        International Journal of Computational Intelligence Systems (2018).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Overrides its parent class with the receiving params
        super(GCO, self).__init__()

        # Cross-ratio
        self.CR = 0.7

        # Mutation factor
        self.F = 1.25

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    @property
    def CR(self):
        """float: Cross-ratio parameter.

        """

        return self._CR

    @CR.setter
    def CR(self, CR):
        if not isinstance(CR, (float, int)):
            raise e.TypeError('`CR` should be a float or integer')
        if CR < 0 or CR > 1:
            raise e.ValueError('`CR` should be between 0 and 1')

        self._CR = CR

    @property
    def F(self):
        """float: Mutation factor.

        """

        return self._F

    @F.setter
    def F(self, F):
        if not isinstance(F, (float, int)):
            raise e.TypeError('`F` should be a float or integer')
        if F < 0:
            raise e.ValueError('`F` should be >= 0')

        self._F = F

    @property
    def life(self):
        """np.array: Array of lives.

        """

        return self._life

    @life.setter
    def life(self, life):
        if not isinstance(life, np.ndarray):
            raise e.TypeError('`life` should be a numpy array')

        self._life = life

    @property
    def counter(self):
        """np.array: Array of counters.

        """

        return self._counter

    @counter.setter
    def counter(self, counter):
        if not isinstance(counter, np.ndarray):
            raise e.TypeError('`counter` should be a numpy array')

        self._counter = counter

    def create_additional_attrs(self, space):
        """Creates additional attributes that are used by this optimizer.

        Args:
            space (Space): A Space object containing meta-information.

        """

        # Array of lives and counters
        self.life = r.generate_uniform_random_number(70, 70, space.n_agents)
        self.counter = np.ones(space.n_agents)

    def _mutate_cell(self, agent, alpha, beta, gamma):
        """Mutates a new cell based on distinct cells (alg. 2).

        Args:
            agent (Agent): Current agent.
            alpha (Agent): 1st picked agent.
            beta (Agent): 2nd picked agent.
            gamma (Agent): 3rd picked agent.

        Returns:
            A mutated cell.

        """

        # Makes a deep copy of agent
        a = copy.deepcopy(agent)

        # For every decision variable
        for j in range(a.n_variables):
            # Generates a second random number
            r2 = r.generate_uniform_random_number()

            # If random number is smaller than cross-ratio
            if r2 < self.CR:
                # Updates the cell position
                a.position[j] = alpha.position[j] + self.F * (beta.position[j] - gamma.position[j])

        return a

    def _dark_zone(self, agents, function):
        """Performs the dark-zone update process (alg. 1).

        Args:
            agents (list): List of agents.
            function (Function): A Function object that will be used as the objective function.

        """

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Generates the first random number, between 0 and 100
            r1 = r.generate_uniform_random_number(0, 100)

            # If random number is smaller than cell's life
            if r1 < self.life[i]:
                # Increases it counter by one
                self.counter[i] += 1

            # If it is not smaller
            else:
                # Resets the counter to one
                self.counter[i] = 1

            # Generates the counting distribution and pick three cells
            C = d.generate_choice_distribution(len(agents), self.counter / np.sum(self.counter), size=3)

            # Mutates a new cell based on current and pre-picked cells
            a = self._mutate_cell(agent, agents[C[0]], agents[C[1]], agents[C[2]])

            # Check agent limits
            a.clip_by_bound()

            # Calculates the fitness for the temporary position
            a.fit = function(a.position)

            # If new fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Copy its position and fitness to the agent
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

                # Increases the life of cell by ten
                self.life[i] += 10

    def _light_zone(self, agents):
        """Performs the light-zone update process (alg. 1).

        Args:
            agents (list): List of agents.

        """

        # Gathers a list of fitness from all agents
        fits = [agent.fit for agent in agents]

        # Calculates the minimum and maximum fitness
        min_fit, max_fit = np.min(fits), np.max(fits)

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Resets the cell life to 10
            self.life[i] = 10

            # Calculates the current cell new life fitness
            life_fit = (agent.fit - max_fit) / (min_fit - max_fit + c.EPSILON)

            # Adds 10 * new life fitness to cell's life
            self.life[i] += 10 * life_fit

    def update(self, space, function):
        """Wraps Germinal Center Optimization over all agents and variables.

        Args:
            space (Space): Space containing agents and update-related information.
            life (np.array): An array holding each cell's current life.
            counter (np.array): An array holding each cell's copy counter.

        """

        # Performs the dark-zone update process
        self._dark_zone(space.agents, function)

        # Performs the light-zone update process
        self._light_zone(space.agents)

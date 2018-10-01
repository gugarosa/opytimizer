import numpy as np
import opytimizer.utils.logging as l
import opytimizer.utils.random as r

from opytimizer.core.agent import Agent

logger = l.get_logger(__name__)


class Space:
    """
    """

    def __init__(self, n_agents=1, n_variables=2, n_dimensions=1, n_iterations=10):
        """
        """

        logger.info('Initializing class: Space')

        # Space useful variables
        self.n_agents = n_agents
        self.n_variables = n_variables
        self.n_dimensions = n_dimensions
        self.n_iterations = n_iterations

        # Space's agents variables
        self.agents = []
        self.best_agent = None

        # Space's bounds variables
        self.lb = np.zeros(n_variables)
        self.ub = np.ones(n_variables)

        # Internal use variables
        self._built = False

        # Creating space's agents
        self._create_agents()

        # We will log some important information
        logger.info('Space created with: ' + str(n_agents) +
                    ' agents and ' + str(n_iterations) + ' iterations')

    def _create_agents(self):
        """
        """

        logger.debug('Running private method: _create_agents()')

        # Iterate through number of agents
        for i in range(self.n_agents):
            # Appends new agent to list
            self.agents.append(
                Agent(n_variables=self.n_variables, n_dimensions=self.n_dimensions))

        # Apply a random agent as the best one
        self.best_agent = self.agents[0]

        logger.debug('Agents were created.')

    def _initialize_agents(self):
        """
        """

        logger.debug('Running private method: _initialize_agents()')

        for agent in self.agents:
            for var in range(self.n_variables):
                agent.position[var] = r.generate_uniform_random_number(
                    self.lb[var], self.ub[var], size=self.n_dimensions)

        logger.debug('Agents were initialized.')

    def _check_bound_size(self, bound, size):
        """
        """

        logger.debug('Running private method: _check_bound_size()')

        if len(bound) != size:
            e = 'Bound needs to be the same size of number of variables.'
            logger.error(e)
            raise RuntimeError(e)
        else:
            logger.debug('Bound was checked without any errors.')
            return True

    def build(self, lower_bound=None, upper_bound=None):
        """
        """

        logger.debug('Running method: build()')

        # Checking if lower bound is avaliable
        if lower_bound:
            # We also need to check if its size matches to our
            # actual number of variables
            if self._check_bound_size(lower_bound, self.n_variables):
                self.lb = lower_bound

        # Checking if upper bound is avaliable
        if upper_bound:
            # We also need to check if its size matches to our
            # actual number of variables
            if self._check_bound_size(upper_bound, self.n_variables):
                self.ub = upper_bound

        # As now we can assume the bounds are correct, we need to
        # initialize our agents
        self._initialize_agents()

        # If no errors were shown, we can declared the Space as built
        self._built = True

        logger.debug('Space was successfully built.')

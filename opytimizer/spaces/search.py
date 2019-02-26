import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.logging as l
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space

logger = l.get_logger(__name__)


class SearchSpace(Space):
    """A SearchSpace class that will hold agents, variables and methods
    related to the search space.

    Methods:
        _initialize_agents(agents, lower_bound, upper_bound): Initialize the Space's agents,
        setting random numbers [lower_bound, upper_bound] to their position.

    """

    def __init__(self, n_agents=1, n_variables=2, n_iterations=10, lower_bound=None, upper_bound=None):
        """Initialization method.

        Args:
            n_agents (int): Number of agents.
            n_variables (int): Number of decision variables.
            n_iterations (int): Number of iterations.
            lower_bound (np.array): Lower bound array with the minimum possible values.
            upper_bound (np.array): Upper bound array with the maximum possible values.

        """

        logger.info('Overriding class: Space -> SearchSpace.')

        # Override its parent class with the receiving arguments
        super(SearchSpace, self).__init__(n_agents=n_agents, n_variables=n_variables,
                                         n_iterations=n_iterations, lower_bound=lower_bound, upper_bound=upper_bound)

        # Now, we need to build this class up
        self._build(lower_bound, upper_bound)

        # Initializing agents
        self._initialize_agents(self.agents, self.lb, self.ub)

        # We will log some important information
        logger.info('Class overrided.')

    def _initialize_agents(self, agents, lower_bound, upper_bound):
        """Initialize agents' position array with uniform random numbers.

        Args:
            agents (list): List of agents.
            lower_bound (np.array): Lower bound array with the minimum possible values.
            upper_bound (np.array): Upper bound array with the maximum possible values.

        """

        logger.debug('Running private method: initialize_agents().')

        # Iterate through all agents
        for agent in agents:
            # Iterate through all decision variables
            for j, (lb, ub) in enumerate(zip(lower_bound, upper_bound)):
                # For each decision variable, we generate uniform random numbers
                agent.position[j] = r.generate_uniform_random_number(
                    lb, ub, size=agent.n_dimensions)

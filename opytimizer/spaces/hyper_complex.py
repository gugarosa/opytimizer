"""Hypercomplex-based search space.
"""

import copy

import numpy as np

import opytimizer.utils.logging as l
from opytimizer.core import Space

logger = l.get_logger(__name__)


class HyperComplexSpace(Space):
    """An HyperComplexSpace class that will hold agents, variables and methods
    related to the hypercomplex search space.

    """

    def __init__(self, n_agents, n_variables, n_dimensions):
        """Initialization method.

        Args:
            n_agents (int): Number of agents.
            n_variables (int): Number of decision variables.
            n_dimensions (int): Number of search space dimensions.

        """

        logger.info('Overriding class: Space -> HyperComplexSpace.')

        # Defines missing override arguments
        lower_bound = np.zeros(n_variables)
        upper_bound = np.ones(n_variables)

        # Overrides its parent class with the receiving arguments
        super(HyperComplexSpace, self).__init__(n_agents, n_variables, n_dimensions,
                                                lower_bound, upper_bound)

        # Builds the class
        self.build()

        logger.info('Class overrided.')

    def _initialize_agents(self):
        """Initializes agents with their positions and defines a best agent.

        """

        # Iterates through all agents
        for agent in self.agents:
            # Initializes the agent
            agent.fill_with_uniform()

        # Defines a best agent
        self.best_agent = copy.deepcopy(self.agents[0])

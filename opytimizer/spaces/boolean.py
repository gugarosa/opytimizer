"""Boolean-based search space.
"""

import copy

import numpy as np

import opytimizer.utils.logging as l
from opytimizer.core.space import Space

logger = l.get_logger(__name__)


class BooleanSpace(Space):
    """A BooleanSpace class for agents, variables and methods
    related to the boolean search space.

    """

    def __init__(self, n_agents, n_variables):
        """Initialization method.

        Args:
            n_agents (int): Number of agents.
            n_variables (int): Number of decision variables.

        """

        logger.info('Overriding class: Space -> BooleanSpace.')

        # Defines missing override arguments
        n_dimensions = 1
        lower_bound = np.zeros(n_variables)
        upper_bound = np.ones(n_variables)

        # Overrides its parent class with the receiving arguments
        super(BooleanSpace, self).__init__(n_agents, n_variables, n_dimensions,
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
            agent.fill_with_binary()

        # Defines a best agent
        self.best_agent = copy.deepcopy(self.agents[0])

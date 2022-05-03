"""Traditional-based search space.
"""

import copy

from opytimizer.core import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class SearchSpace(Space):
    """A SearchSpace class for agents, variables and methods
    related to the search space.

    """

    def __init__(self, n_agents, n_variables, lower_bound, upper_bound):
        """Initialization method.

        Args:
            n_agents (int): Number of agents.
            n_variables (int): Number of decision variables.
            lower_bound (float, list, tuple, np.array): Minimum possible values.
            upper_bound (float, list, tuple, np.array): Maximum possible values.

        """

        logger.info("Overriding class: Space -> SearchSpace.")

        # Defines missing override arguments
        n_dimensions = 1

        super(SearchSpace, self).__init__(
            n_agents, n_variables, n_dimensions, lower_bound, upper_bound
        )

        self.build()

        logger.info("Class overrided.")

    def _initialize_agents(self):
        """Initializes agents with their positions and defines a best agent."""

        for agent in self.agents:
            agent.fill_with_uniform()

        self.best_agent = copy.deepcopy(self.agents[0])

"""Hypercomplex-based search space.
"""

import copy

import numpy as np

from opytimizer.core import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class HyperComplexSpace(Space):
    """An HyperComplexSpace class that will hold agents, variables and methods
    related to the hypercomplex search space.

    """

    def __init__(self, n_agents: int, n_variables: int, n_dimensions: int) -> None:
        """Initialization method.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_dimensions: Number of search space dimensions.

        """

        logger.info("Overriding class: Space -> HyperComplexSpace.")

        # Defines missing override arguments
        lower_bound = np.zeros(n_variables)
        upper_bound = np.ones(n_variables)

        super(HyperComplexSpace, self).__init__(
            n_agents, n_variables, n_dimensions, lower_bound, upper_bound
        )

        self.build()

        logger.info("Class overrided.")

    def _initialize_agents(self) -> None:
        """Initializes agents with their positions and defines a best agent."""

        for agent in self.agents:
            agent.fill_with_uniform()

        self.best_agent = copy.deepcopy(self.agents[0])

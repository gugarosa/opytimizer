"""Search space."""

from typing import List, Optional, Tuple, Union

import numpy as np

from opytimizer.core.agent import Agent


class Space:
    """A Space class for agents, variables and methods
    related to the search space.

    """

    def __init__(
        self,
        n_agents: int = 1,
        n_variables: int = 1,
        n_dimensions: int = 1,
        lower_bound: Optional[Union[float, List, Tuple, np.ndarray]] = 0.0,
        upper_bound: Optional[Union[float, List, Tuple, np.ndarray]] = 1.0,
        mapping: Optional[List[str]] = None,
    ) -> None:
        """Initialization method.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_dimensions: Dimension of search space.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
            mapping: String-based identifiers for mapping variables' names.

        """

        if not isinstance(n_agents, int):
            raise TypeError("`n_agents` should be an integer")
        if n_agents <= 0:
            raise ValueError("`n_agents` should be > 0")

        best_agent = Agent(n_variables, n_dimensions, lower_bound, upper_bound, mapping)

        self.n_agents = n_agents
        self.n_variables = best_agent.n_variables
        self.n_dimensions = best_agent.n_dimensions
        self.lb = best_agent.lb
        self.ub = best_agent.ub
        self.mapping = best_agent.mapping

        self.agents = []
        self.best_agent = best_agent

    def _create_agents(self) -> None:
        """Creates a list of agents."""

        self.agents = [
            Agent(self.n_variables, self.n_dimensions, self.lb, self.ub, self.mapping)
            for _ in range(self.n_agents)
        ]

    def _initialize_agents(self) -> None:
        """Initializes agents with their positions and defines a best agent.

        As each child has a different procedure of initialization,
        you will need to implement it directly on its class.

        """

        pass

    def build(self) -> None:
        """Builds the object by creating and initializing the agents."""

        self._create_agents()
        self._initialize_agents()

    def clip_by_bound(self) -> None:
        """Clips the agents' decision variables to the bounds limits."""

        for agent in self.agents:
            agent.clip_by_bound()

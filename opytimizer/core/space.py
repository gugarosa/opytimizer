"""Search space."""

from typing import List, Optional, Tuple, Union

import numpy as np

from opytimizer.core.agent import Agent


class Space:
    """Own population configuration, candidate agents, and the best-agent state.

    The base constructor configures the space without populating ``agents``.
    ``build`` creates the population and invokes the initializer hook. Concrete
    spaces normally call it from their constructors.

    Attributes:
        agents: Mutable candidate population. Preserve each position's
            ``(n_variables, n_dimensions)`` shape when updating it.
        best_agent: Independent best-position/fitness state maintained by
            initialization and optimizer evaluation.
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
        """Configure population dimensions and bounds without creating candidates.

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
        """Initialize candidate positions and best-agent state in a subclass.

        The base hook does nothing, leaving the newly created zero positions.
        """

        pass

    def build(self) -> None:
        """Replace the population and invoke the space-specific initializer."""

        self._create_agents()
        self._initialize_agents()

    def clip_by_bound(self) -> None:
        """Clips the agents' decision variables to the bounds limits."""

        for agent in self.agents:
            agent.clip_by_bound()

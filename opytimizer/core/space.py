# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Search space.

"""

from numpy.typing import ArrayLike

from opytimizer.core.agent import Agent


class Space:
    """Own population configuration, candidate agents, and the best-agent state.

    """

    def __init__(
        self,
        n_agents: int = 1,
        n_variables: int = 1,
        n_dimensions: int = 1,
        lower_bound: ArrayLike | None = 0.0,
        upper_bound: ArrayLike | None = 1.0,
        mapping: list[str] | None = None,
    ) -> None:
        """Configure population dimensions and bounds without creating candidates.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_dimensions: Dimension of search space.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
            mapping: String-based identifiers for mapping variables' names.

        Raises:
            TypeError: Population dimensions or the supplied mapping have invalid types.
            ValueError: Dimensions are not positive or bounds and mapping do not match the variable count.

        Notes:
            ``agents`` starts empty. ``build`` replaces it and invokes the
            initializer hook, which concrete spaces normally call during
            construction. The base initializer leaves zero positions unchanged.
            Keep each mutable position's ``(n_variables, n_dimensions)`` shape.
            Initialization and optimizer evaluation maintain an independent
            ``best_agent`` position and fitness. Bounds and mapping share the
            supplied arrays and list. ``None`` bounds are retained for base-space
            configuration, not for numeric clipping or uniform initialization.

        """

        if not isinstance(n_agents, int):
            raise TypeError("`n_agents` should be an integer.")
        if n_agents <= 0:
            raise ValueError("`n_agents` should be > 0.")

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
        self.agents = [
            Agent(self.n_variables, self.n_dimensions, self.lb, self.ub, self.mapping) for _ in range(self.n_agents)
        ]

    def _initialize_agents(self) -> None:
        pass

    def build(self) -> None:
        """Replace the population and invoke the space-specific initializer.

        """

        self._create_agents()
        self._initialize_agents()

    def clip_by_bound(self) -> None:
        """Clips the agents' decision variables to the bounds limits.

        """

        for agent in self.agents:
            agent.clip_by_bound()

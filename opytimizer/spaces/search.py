# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Traditional-based search space.

"""

import copy

from numpy.typing import ArrayLike

from opytimizer.core import Space


class SearchSpace(Space):
    """Own a population of bounded real-valued decision variables.

    """

    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        lower_bound: ArrayLike,
        upper_bound: ArrayLike,
        mapping: list[str] | None = None,
    ) -> None:
        """Build uniformly bounded positions and an independent copy of the first best agent.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
            mapping: String-based identifiers for mapping variables' names.

        """

        n_dimensions = 1

        super().__init__(n_agents, n_variables, n_dimensions, lower_bound, upper_bound, mapping)

        self.build()

    def _initialize_agents(self) -> None:
        for agent in self.agents:
            agent.fill_with_uniform()

        self.best_agent = copy.deepcopy(self.agents[0])

# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Hypercomplex-based search space.

"""

import copy

import numpy as np

from opytimizer.core import Space


class HyperComplexSpace(Space):
    """Own a population of hypercomplex decision variables.

    """

    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        n_dimensions: int,
        mapping: list[str] | None = None,
    ) -> None:
        """Build uniform positions in the unit interval and copy the first best agent.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            n_dimensions: Number of search space dimensions.
            mapping: String-based identifiers for mapping variables' names.

        """

        lower_bound = np.zeros(n_variables)
        upper_bound = np.ones(n_variables)

        super().__init__(n_agents, n_variables, n_dimensions, lower_bound, upper_bound, mapping)

        self.build()

    def _initialize_agents(self) -> None:
        for agent in self.agents:
            agent.fill_with_uniform()

        self.best_agent = copy.deepcopy(self.agents[0])

# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Boolean-based search space.

"""

import copy

import numpy as np

from opytimizer.core import Space


class BooleanSpace(Space):
    """Own a population of binary decision variables.

    """

    def __init__(self, n_agents: int, n_variables: int, mapping: list[str] | None = None) -> None:
        """Build binary positions and an independent copy of the first best agent.

        Args:
            n_agents: Number of agents.
            n_variables: Number of decision variables.
            mapping: String-based identifiers for mapping variables' names.

        """

        n_dimensions = 1
        lower_bound = np.zeros(n_variables)
        upper_bound = np.ones(n_variables)

        super().__init__(n_agents, n_variables, n_dimensions, lower_bound, upper_bound, mapping)

        self.build()

    def _initialize_agents(self) -> None:
        for agent in self.agents:
            agent.fill_with_binary()

        self.best_agent = copy.deepcopy(self.agents[0])

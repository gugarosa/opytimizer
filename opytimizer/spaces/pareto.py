# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Pareto-based search space.

"""

import copy

import numpy as np

from opytimizer.core import Space


class ParetoSpace(Space):
    """Own candidate points for Pareto-frontier evaluation without clipping them.

    """

    def __init__(self, data_points: np.ndarray, mapping: list[str] | None = None) -> None:
        """Load data points into agents and copy the first agent as the best state.

        Args:
            data_points: Non-empty matrix of rows retained as views by agent positions.
            mapping: String-based identifiers for mapping variables' names.

        """

        if not isinstance(data_points, np.ndarray):
            raise TypeError("`data_points` should be a numpy array.")
        if data_points.ndim != 2 or not all(data_points.shape):
            raise ValueError("`data_points` should be a non-empty matrix.")

        n_agents, n_variables = data_points.shape
        n_dimensions = 1
        lower_bound = [0] * n_variables
        upper_bound = [0] * n_variables

        super().__init__(n_agents, n_variables, n_dimensions, lower_bound, upper_bound, mapping)

        self.build(data_points)

    def _load_agents(self, data_points: np.ndarray) -> None:
        for agent, data in zip(self.agents, data_points):
            agent.position = np.expand_dims(data, -1)

        self.best_agent = copy.deepcopy(self.agents[0])

    def build(self, data_points: np.ndarray) -> None:
        """Replace the population with views of the supplied data rows.

        Args:
            data_points: Matrix matching the configured population and variable counts.

        Notes:
            Agent positions share the corresponding input rows. The best agent
            is an independent copy of the first loaded agent. Bounds clipping
            is disabled because Pareto candidates are not bounded search values.

        """

        self._create_agents()
        self._load_agents(data_points)

    def clip_by_bound(self) -> None:
        pass

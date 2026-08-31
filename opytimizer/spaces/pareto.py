"""Pareto-based search space."""

import copy
from typing import List, Optional

import numpy as np

from opytimizer.core import Space


class ParetoSpace(Space):
    """A ParetoSpace class for agents, variables and methods
    related to the pareto-frontier search space.

    """

    def __init__(
        self, data_points: np.ndarray, mapping: Optional[List[str]] = None
    ) -> None:
        """Initialization method.

        Args:
            data_points: Pre-defined data points.
            mapping: String-based identifiers for mapping variables' names.

        """

        if not isinstance(data_points, np.ndarray):
            raise TypeError("`data_points` should be a numpy array")
        if data_points.ndim != 2 or not all(data_points.shape):
            raise ValueError("`data_points` should be a non-empty matrix")

        n_agents, n_variables = data_points.shape
        n_dimensions = 1
        lower_bound = [0] * n_variables
        upper_bound = [0] * n_variables

        super().__init__(
            n_agents, n_variables, n_dimensions, lower_bound, upper_bound, mapping
        )

        self.build(data_points)

    def _load_agents(self, data_points: np.ndarray) -> None:
        """Loads agents from pre-defined data points.

        Args:
            data_points: Pre-defined data points.

        """

        for agent, data in zip(self.agents, data_points):
            agent.position = np.expand_dims(data, -1)

        self.best_agent = copy.deepcopy(self.agents[0])

    def build(self, data_points: np.ndarray) -> None:
        """Builds the object by creating and pre-loading the agents.

        Args:
            data_points: Pre-defined data points.

        """

        self._create_agents()
        self._load_agents(data_points)

    def clip_by_bound(self) -> None:
        """Overrides default function as no clipping should be performed."""

        pass

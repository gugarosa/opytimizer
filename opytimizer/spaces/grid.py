# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Grid-based search space.

"""

import copy

import numpy as np
from numpy.typing import ArrayLike

from opytimizer.core import Space


class GridSpace(Space):
    """Own the Cartesian grid of bounded decision-variable values.

    """

    def __init__(
        self,
        n_variables: int,
        step: ArrayLike,
        lower_bound: ArrayLike,
        upper_bound: ArrayLike,
        mapping: list[str] | None = None,
    ) -> None:
        """Build one agent per grid point and copy the first agent as the best state.

        Args:
            n_variables: Number of decision variables.
            step: Positive, finite spacing for each variable.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values, included when reached by a step.
            mapping: String-based identifiers for mapping variables' names.

        """

        n_agents = 1
        n_dimensions = 1

        super().__init__(n_agents, n_variables, n_dimensions, lower_bound, upper_bound, mapping)

        step = np.asarray(step)
        if not step.shape:
            step = np.expand_dims(step, -1)
        if step.shape != (self.n_variables,):
            raise ValueError("`step` should match `n_variables`.")
        if not np.all(np.isfinite(step)) or np.any(step <= 0):
            raise ValueError("`step` should contain finite values > 0.")
        if self.lb.ndim != 1 or self.ub.ndim != 1:
            raise ValueError("`lower_bound` and `upper_bound` should be one-dimensional.")
        if not np.all(np.isfinite(self.lb)) or not np.all(np.isfinite(self.ub)) or np.any(self.lb > self.ub):
            raise ValueError("`lower_bound` and `upper_bound` should be finite and lower <= upper.")
        self.step = step

        self._create_grid()
        self.build()

    def _create_grid(self) -> None:
        axes = []
        for s, lb, ub in zip(self.step, self.lb, self.ub):
            lb, ub = float(lb), float(ub)
            n_steps = int(np.ceil((ub - lb) / s))
            values = lb + s * np.arange(n_steps + 1, dtype=float)
            # Retain rounded endpoints, but not a full step beyond the bounds
            tolerance = min(s / 2, 2 * np.spacing(max(abs(lb), abs(ub))))
            values = values[(values <= ub) | (np.abs(values - ub) <= tolerance)]
            axes.append(np.minimum(values, ub))

        mesh = np.meshgrid(*axes)

        self.grid = np.array(([m.ravel() for m in mesh])).T
        self.n_agents = len(self.grid)

    def _initialize_agents(self) -> None:
        for agent, grid in zip(self.agents, self.grid):
            agent.fill_with_static(grid)

        self.best_agent = copy.deepcopy(self.agents[0])

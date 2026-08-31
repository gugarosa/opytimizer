"""Grid-based search space."""

import copy
from typing import List, Optional, Tuple, Union

import numpy as np

from opytimizer.core import Space


class GridSpace(Space):
    """A GridSpace class for agents, variables and methods
    related to the grid search space.

    """

    def __init__(
        self,
        n_variables: int,
        step: Union[float, List, Tuple, np.ndarray],
        lower_bound: Union[float, List, Tuple, np.ndarray],
        upper_bound: Union[float, List, Tuple, np.ndarray],
        mapping: Optional[List[str]] = None,
    ) -> None:
        """Initialization method.

        Args:
            n_variables: Number of decision variables.
            step: Variables' steps.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
            mapping: String-based identifiers for mapping variables' names.

        """

        n_agents = 1
        n_dimensions = 1

        super().__init__(
            n_agents, n_variables, n_dimensions, lower_bound, upper_bound, mapping
        )

        step = np.asarray(step)
        if not step.shape:
            step = np.expand_dims(step, -1)
        if step.shape[0] != self.n_variables:
            raise ValueError("`step` should match `n_variables`")
        self.step = step

        self._create_grid()
        self.build()

    def _create_grid(self) -> None:
        """Creates a grid of possible search values."""

        mesh = np.meshgrid(
            *[
                s * np.arange(lb / s, ub / s + s)
                for s, lb, ub in zip(self.step, self.lb, self.ub)
            ]
        )

        self.grid = np.array(([m.ravel() for m in mesh])).T
        self.n_agents = len(self.grid)

    def _initialize_agents(self) -> None:
        """Initializes agents with their positions and defines a best agent."""

        for agent, grid in zip(self.agents, self.grid):
            agent.fill_with_static(grid)

        self.best_agent = copy.deepcopy(self.agents[0])

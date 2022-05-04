"""Grid-based search space.
"""

import copy
from typing import List, Tuple, Union

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


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
    ) -> None:
        """Initialization method.

        Args:
            n_variables: Number of decision variables.
            step: Variables' steps.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.

        """

        logger.info("Overriding class: Space -> GridSpace.")

        # Defines missing override arguments
        # `n_agents = 1` is used as a placeholder for now
        n_agents = 1
        n_dimensions = 1

        super(GridSpace, self).__init__(
            n_agents, n_variables, n_dimensions, lower_bound, upper_bound
        )

        # Step size of each variable
        self.step = np.asarray(step)

        self._create_grid()

        self.build()

        logger.info("Class overrided.")

    @property
    def step(self) -> np.ndarray:
        """Step size of each variable."""

        return self._step

    @step.setter
    def step(self, step: np.ndarray) -> None:
        if not isinstance(step, np.ndarray):
            raise e.TypeError("`step` should be a numpy array")
        if not step.shape:
            step = np.expand_dims(step, -1)
        if step.shape[0] != self.n_variables:
            raise e.SizeError("`step` should be the same size as `n_variables`")

        self._step = step

    @property
    def grid(self) -> np.ndarray:
        """Grid with possible search values."""

        return self._grid

    @grid.setter
    def grid(self, grid: np.ndarray) -> None:
        if not isinstance(grid, np.ndarray):
            raise e.TypeError("`grid` should be a numpy array")

        self._grid = grid

    def _create_grid(self) -> None:
        """Creates a grid of possible search values."""

        # Creates a meshgrid with all possible search values
        mesh = np.meshgrid(
            *[
                s * np.arange(lb / s, ub / s + s)
                for s, lb, ub in zip(self.step, self.lb, self.ub)
            ]
        )

        # Transforms the meshgrid into a list
        # and re-defines the number of agents to the length of grid
        self.grid = np.array(([m.ravel() for m in mesh])).T
        self.n_agents = len(self.grid)

    def _initialize_agents(self) -> None:
        """Initializes agents with their positions and defines a best agent."""

        for agent, grid in zip(self.agents, self.grid):
            agent.fill_with_static(grid)

        self.best_agent = copy.deepcopy(self.agents[0])

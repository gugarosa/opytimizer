"""Grid-based search space.
"""

import copy

import numpy as np

import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core import Space

logger = l.get_logger(__name__)


class GridSpace(Space):
    """A GridSpace class for agents, variables and methods
    related to the grid search space.

    """

    def __init__(self, n_variables, step, lower_bound, upper_bound):
        """Initialization method.

        Args:
            n_variables (int): Number of decision variables.
            step (float, list, tuple, np.array): Variables' steps.
            lower_bound (float, list, tuple, np.array): Minimum possible values.
            upper_bound (float, list, tuple, np.array): Maximum possible values.

        """

        logger.info('Overriding class: Space -> GridSpace.')

        # Defines missing override arguments
        # `n_agents = 1` is used as a placeholder for now
        n_agents = 1
        n_dimensions = 1

        super(GridSpace, self).__init__(n_agents, n_variables, n_dimensions,
                                        lower_bound, upper_bound)

        # Step size of each variable
        self.step = np.asarray(step)

        self._create_grid()

        self.build()

        logger.info('Class overrided.')

    @property
    def step(self):
        """np.array: Step size of each variable.

        """

        return self._step

    @step.setter
    def step(self, step):
        if not isinstance(step, np.ndarray):
            raise e.TypeError('`step` should be a numpy array')
        if not step.shape:
            step = np.expand_dims(step, -1)
        if step.shape[0] != self.n_variables:
            raise e.SizeError('`step` should be the same size as `n_variables`')

        self._step = step

    @property
    def grid(self):
        """np.array: Grid with possible search values.

        """

        return self._grid

    @grid.setter
    def grid(self, grid):
        if not isinstance(grid, np.ndarray):
            raise e.TypeError('`grid` should be a numpy array')

        self._grid = grid

    def _create_grid(self):
        """Creates a grid of possible search values.

        """

        # Creates a meshgrid with all possible search values
        mesh = np.meshgrid(*[s * np.arange(lb / s, ub / s + s)
                             for s, lb, ub in zip(self.step, self.lb, self.ub)])

        # Transforms the meshgrid into a list
        # and re-defines the number of agents to the length of grid
        self.grid = np.array(([m.ravel() for m in mesh])).T
        self.n_agents = len(self.grid)

    def _initialize_agents(self):
        """Initializes agents with their positions and defines a best agent.

        """

        for agent, grid in zip(self.agents, self.grid):
            agent.fill_with_static(grid)

        self.best_agent = copy.deepcopy(self.agents[0])

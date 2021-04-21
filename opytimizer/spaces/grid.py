"""Grid-based search space.
"""

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core.space import Space

logger = l.get_logger(__name__)


class GridSpace(Space):
    """A GridSpace class for agents, variables and methods
    related to the grid search space.

    """

    def __init__(self, n_variables=1, step=(0.1,), lower_bound=(0,), upper_bound=(1,)):
        """Initialization method.

        Args:
            n_variables (int): Number of decision variables.
            step (tuple): Size of each variable step in the grid.
            lower_bound (tuple): Lower bound tuple with the minimum possible values.
            upper_bound (tuple): Upper bound tuple with the maximum possible values.

        """

        logger.info('Overriding class: Space -> GridSpace.')

        # Defining a property to hold the step size
        self.step = step

        # Creating the searching grid
        self._create_grid(step, lower_bound, upper_bound)

        # Override its parent class with the receiving arguments
        super(GridSpace, self).__init__(len(self.grid), n_variables, n_iterations=1)

        # Now, we need to build this class up
        self._build(lower_bound, upper_bound)

        # Initializing agents
        self._initialize_agents()

        logger.info('Class overrided.')

    @property
    def step(self):
        """tuple: Size of each variable step.

        """

        return self._step

    @step.setter
    def step(self, step):
        if not isinstance(step, tuple):
            raise e.TypeError('`step` should be a tuple')

        self._step = step

    @property
    def grid(self):
        """list: Grid with possible searching values.

        """

        return self._grid

    @grid.setter
    def grid(self, grid):
        if not isinstance(grid, np.ndarray):
            raise e.TypeError('`grid` should be a numpy array')

        self._grid = grid

    def _create_grid(self, step, lower_bound, upper_bound):
        """Creates a grid of possible searches.

        Args:
            step (tuple): Size of each variable step in the grid.
            lower_bound (tuple): Lower bound tuple with the minimum possible values.
            upper_bound (tuple): Upper bound tuple with the maximum possible values.

        """

        logger.debug('Running private method: create_grid().')

        # Checks if number of steps equals the number of lower and upper bounds
        if len(step) != len(lower_bound) or len(step) != len(upper_bound):
            # If not, raises an error
            raise e.SizeError('`step` should have the same size of `lower_bound` and `upper_bound`')

        # Creating a meshgrid with all possible searches
        mesh = np.meshgrid(*[s * np.arange(lb / s, ub / s + s)
                             for s, lb, ub in zip(step, lower_bound, upper_bound)])

        # Transforming the meshgrid into a list of possible searches
        self.grid = np.array(([m.ravel() for m in mesh])).T

        logger.debug('Grid created with step size equal to %s.', step)

    def _initialize_agents(self):
        """Initialize agents' position array with grid values.

        """

        logger.debug('Running private method: initialize_agents().')

        # Iterates through all agents and grid options
        for agent, grid in zip(self.agents, self.grid):
            # Iterates through all decision variables
            for j, (lb, ub, g) in enumerate(zip(self.lb, self.ub, grid)):
                # For each decision variable, we use the grid values
                agent.position[j] = r.generate_uniform_random_number(g, g, agent.n_dimensions)

                # Applies the lower and upper bounds
                agent.lb[j] = lb
                agent.ub[j] = ub

        logger.debug('Agents initialized.')

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core.space import Space

logger = l.get_logger(__name__)


class GridSpace(Space):
    """A GridSpace class for agents, variables and methods
    related to the grid space.

    """

    def __init__(self, n_variables=1, step=0.1, lower_bound=[0], upper_bound=[1]):
        """Initialization method.

        Args:
            n_variables (int): Number of decision variables.
            step (float): Size of each step in the grid.
            lower_bound (list): Lower bound list with the minimum possible values.
            upper_bound (list): Upper bound list with the maximum possible values.

        """

        logger.info('Overriding class: Space -> GridSpace.')

        # Defining a property to hold the step size
        self.step = step

        # Creating the searching grid
        n_agents = self._create_grid(step, lower_bound, upper_bound)

        # Override its parent class with the receiving arguments
        super(GridSpace, self).__init__(n_agents, n_variables, n_iterations=1)

        # Now, we need to build this class up
        self._build(lower_bound, upper_bound)

        # Initializing agents
        self._initialize_agents()

        logger.info('Class overrided.')

    @property
    def step(self):
        """float: Size of each possible step.

        """

        return self._step

    @step.setter
    def step(self, step):
        if not (isinstance(step, float) or isinstance(step, int)):
            raise e.TypeError('`step` should be a float or integer')
        if step <= 0:
            raise e.ValueError('`step` should be > 0')

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
            step (float): Size of each step in the grid.
            lower_bound (list): Lower bound list with the minimum possible values.
            upper_bound (list): Upper bound list with the maximum possible values.

        Returns:
            The number of possible searches.

        """

        logger.debug('Running private method: create_grid().')

        # Creating a meshgrid with all possible search options
        mesh = np.meshgrid(*[step * np.arange(lb / step, ub / step)
                             for lb, ub in zip(lower_bound, upper_bound)])

        # Transforming the meshgrid into a list of possible searches
        self.grid = np.array(([m.ravel() for m in mesh])).T

        logger.debug(f'Grid created with step size equal to {step}.')

        return len(self.grid)

    def _initialize_agents(self):
        """Initialize agents' position array with uniform random numbers.

        """

        logger.debug('Running private method: initialize_agents().')

        # Iterate through all agents and grid options
        for agent, grid in zip(self.agents, self.grid):
            # Iterate through all decision variables
            for j, (lb, ub, g) in enumerate(zip(self.lb, self.ub, grid)):
                # For each decision variable, we generate uniform random numbers
                agent.position[j] = r.generate_uniform_random_number(g, g, agent.n_dimensions)

                # Applies the lower bound the agent's lower bound
                agent.lb[j] = lb

                # And also the upper bound
                agent.ub[j] = ub

        logger.debug('Agents initialized.')

"""Search space.
"""

import numpy as np

import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core import Agent

logger = l.get_logger(__name__)


class Space:
    """A Space class for agents, variables and methods
    related to the search space.

    """

    def __init__(self, n_agents=1, n_variables=1, n_dimensions=1, lower_bound=0.0, upper_bound=1.0):
        """Initialization method.

        Args:
            n_agents (int): Number of agents.
            n_variables (int): Number of decision variables.
            n_dimensions (int): Dimension of search space.
            lower_bound (float, list, tuple, np.array): Minimum possible values.
            upper_bound (float, list, tuple, np.array): Maximum possible values.

        """

        # Number of agents
        self.n_agents = n_agents

        # Number of variables
        self.n_variables = n_variables

        # Number of dimensions
        self.n_dimensions = n_dimensions

        # Lower bounds
        self.lb = np.asarray(lower_bound)

        # Upper bounds
        self.ub = np.asarray(upper_bound)

        # Agents
        self.agents = []

        # Best agent
        self.best_agent = Agent(n_variables, n_dimensions, lower_bound, upper_bound)

        # Indicates whether the space is built or not
        self.built = False

    @property
    def n_agents(self):
        """int: Number of agents.

        """

        return self._n_agents

    @n_agents.setter
    def n_agents(self, n_agents):
        if not isinstance(n_agents, int):
            raise e.TypeError('`n_agents` should be an integer')
        if n_agents <= 0:
            raise e.ValueError('`n_agents` should be > 0')

        self._n_agents = n_agents

    @property
    def n_variables(self):
        """int: Number of decision variables.

        """

        return self._n_variables

    @n_variables.setter
    def n_variables(self, n_variables):
        if not isinstance(n_variables, int):
            raise e.TypeError('`n_variables` should be an integer')
        if n_variables <= 0:
            raise e.ValueError('`n_variables` should be > 0')

        self._n_variables = n_variables

    @property
    def n_dimensions(self):
        """int: Number of search space dimensions.

        """

        return self._n_dimensions

    @n_dimensions.setter
    def n_dimensions(self, n_dimensions):
        if not isinstance(n_dimensions, int):
            raise e.TypeError('`n_dimensions` should be an integer')
        if n_dimensions <= 0:
            raise e.ValueError('`n_dimensions` should be > 0')

        self._n_dimensions = n_dimensions

    @property
    def lb(self):
        """np.array: Minimum possible values.

        """

        return self._lb

    @lb.setter
    def lb(self, lb):
        if not isinstance(lb, np.ndarray):
            raise e.TypeError('`lb` should be a numpy array')
        if not lb.shape:
            lb = np.expand_dims(lb, -1)
        if lb.shape[0] != self.n_variables:
            raise e.SizeError('`lb` should be the same size as `n_variables`')

        self._lb = lb

    @property
    def ub(self):
        """np.array: Maximum possible values.

        """

        return self._ub

    @ub.setter
    def ub(self, ub):
        if not isinstance(ub, np.ndarray):
            raise e.TypeError('`ub` should be a numpy array')
        if not ub.shape:
            ub = np.expand_dims(ub, -1)
        if not ub.shape or ub.shape[0] != self.n_variables:
            raise e.SizeError('`ub` should be the same size as `n_variables`')

        self._ub = ub

    @property
    def agents(self):
        """list: Agents that belongs to the space.

        """

        return self._agents

    @agents.setter
    def agents(self, agents):
        if not isinstance(agents, list):
            raise e.TypeError('`agents` should be a list')

        self._agents = agents

    @property
    def best_agent(self):
        """Agent: Best agent.

        """

        return self._best_agent

    @best_agent.setter
    def best_agent(self, best_agent):
        if not isinstance(best_agent, Agent):
            raise e.TypeError('`best_agent` should be an Agent')

        self._best_agent = best_agent

    @property
    def built(self):
        """bool: Indicates whether the space is built.

        """

        return self._built

    @built.setter
    def built(self, built):
        if not isinstance(built, bool):
            raise e.TypeError('`built` should be a boolean')

        self._built = built

    def _create_agents(self):
        """Creates a list of agents.

        """

        # List of agents
        self.agents = [Agent(self.n_variables, self.n_dimensions,
                             self.lb, self.ub) for _ in range(self.n_agents)]

    def _initialize_agents(self):
        """Initializes agents with their positions and defines a best agent.

        As each child has a different procedure of initialization,
        you will need to implement it directly on its class.

        """

        pass

    def build(self):
        """Builds the object by creating and initializing the agents.

        """

        # Creates the agents
        self._create_agents()

        # Initializes the agents
        self._initialize_agents()

        # If no errors were shown, we can declare the space as `built`
        self.built = True

        # Logs the properties
        logger.debug('Agents: %d | Size: (%d, %d) | '
                     'Lower Bound: %s | Upper Bound: %s | Built: %s.',
                     self.n_agents, self.n_variables, self.n_dimensions,
                     self.lb, self.ub, self.built)

    def clip_by_bound(self):
        """Clips the agents' decision variables to the bounds limits.

        """

        # Iterates through all agents
        for agent in self.agents:
            # Clips its limits
            agent.clip_by_bound()

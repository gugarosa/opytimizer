import copy

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core.agent import Agent

logger = l.get_logger(__name__)


class Space:
    """A Space class for agents, variables and methods
    related to the search space.

    """

    def __init__(self, n_agents=1, n_variables=1, n_dimensions=1, n_iterations=10):
        """Initialization method.

        Args:
            n_agents (int): Number of agents.
            n_variables (int): Number of decision variables.
            n_dimensions (int): Dimension of search space.
            n_iterations (int): Number of iterations.

        """

        # Number of agents
        self.n_agents = n_agents

        # Number of variables
        self.n_variables = n_variables

        # Number of dimensions
        self.n_dimensions = n_dimensions

        # Number of iterations
        self.n_iterations = n_iterations

        # List of agents
        self.agents = []

        # Best agent object
        self.best_agent = Agent()

        # Lower bounds
        self.lb = np.zeros(n_variables)

        # Upper bounds
        self.ub = np.ones(n_variables)

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
        """int: Dimension of search space.

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
    def n_iterations(self):
        """int: Number of iterations.

        """

        return self._n_iterations

    @n_iterations.setter
    def n_iterations(self, n_iterations):
        if not isinstance(n_iterations, int):
            raise e.TypeError('`n_iterations` should be an integer')
        if n_iterations <= 0:
            raise e.ValueError('`n_iterations` should be > 0')

        self._n_iterations = n_iterations

    @property
    def agents(self):
        """list: List of agents that belongs to Space.

        """

        return self._agents

    @agents.setter
    def agents(self, agents):
        if not isinstance(agents, list):
            raise e.TypeError('`agents` should be a list')

        self._agents = agents

    @property
    def best_agent(self):
        """Agent: A best agent object from Agent class.

        """

        return self._best_agent

    @best_agent.setter
    def best_agent(self, best_agent):
        if not isinstance(best_agent, Agent):
            raise e.TypeError('`best_agent` should be an Agent')

        self._best_agent = best_agent

    @property
    def lb(self):
        """np.array: Lower bound array with the minimum possible values.

        """

        return self._lb

    @lb.setter
    def lb(self, lb):
        if not isinstance(lb, np.ndarray):
            raise e.TypeError('`lb` should be a numpy array')
        if lb.shape[0] != self.n_variables:
            raise e.SizeError('`lb` should be the same size as `n_variables`')

        self._lb = lb

    @property
    def ub(self):
        """np.array: Upper bound array with the maximum possible values.

        """

        return self._ub

    @ub.setter
    def ub(self, ub):
        if not isinstance(ub, np.ndarray):
            raise e.TypeError('`ub` should be a numpy array')
        if ub.shape[0] != self.n_variables:
            raise e.SizeError('`ub` should be the same size as `n_variables`')

        self._ub = ub

    @property
    def built(self):
        """bool: A boolean to indicate whether the space is built.

        """

        return self._built

    @built.setter
    def built(self, built):
        self._built = built

    def _create_agents(self):
        """Creates a list of agents and the best agent.

        Also defines a random best agent, only for initialization purposes.

        """

        logger.debug('Running private method: create_agents().')

        # Creating a list of agents
        self.agents = [Agent(self.n_variables, self.n_dimensions) for _ in range(self.n_agents)]

        # Apply the first agent as the best one
        self.best_agent = copy.deepcopy(self.agents[0])

    def _initialize_agents(self):
        """Initialize agents' position array with uniform random numbers.

        As each space child can have a different procedure of initializing agents,
        you will need to implement it directly on child's class.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError

    def _build(self, lower_bound, upper_bound):
        """This method serves as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            lower_bound (list): Lower bound array with the minimum possible values.
            upper_bound (list): Upper bound array with the maximum possible values.

        """

        logger.debug('Running private method: build().')

        # Creating lower bound array from list
        self.lb = np.asarray(lower_bound)

        # Creating upper bound array from list
        self.ub = np.asarray(upper_bound)

        # Creating agents
        self._create_agents()

        # If no errors were shown, we can declared the Space as built
        self.built = True

        # Logging attributes
        logger.debug(
            f'Agents: {self.n_agents} | Size: ({self.n_variables}, {self.n_dimensions}) | '
            f'Iterations: {self.n_iterations} | Lower Bound: {self.lb} | '
            f'Upper Bound: {self.ub} | Built: {self.built}.')

import numpy as np
import opytimizer.utils.logging as l
import opytimizer.utils.random as r

from opytimizer.core.agent import Agent

logger = l.get_logger(__name__)


class Space:
    """A Space class that will hold agents, variables and methods
    related to the search space.

    Properties:
        agents (list): List of agents that belongs to Space.
        best_agent (Agent): A best agent object from Agent class.
        built (bool): A boolean to indicate whether the space is built.
        n_agents (int): Number of agents.
        n_dimensions (int): Dimension of search space.
        n_iterations (int): Number of iterations.
        n_variables (int): Number of decision variables.
        lb (np.array): Lower bound array with the minimum possible values.
        ub (np.array): Upper bound array with the maximum possible values.

    Methods:
        __create_agents(): Creates a list of agents.
        __initialize_agents(): Initialize the Space agents, setting random numbers
        to their position.
        __check_bound_size(bound, size): Checks whether the bound's length
        is equal to size parameter.
        _build(lower_bound, upper_bound): An object building method.

    """

    def __init__(self, n_agents=1, n_iterations=10, n_variables=2, n_dimensions=1, lower_bound=None, upper_bound=None):
        """Initialization method.

        Args:
            n_agents (int): Number of agents.
            n_variables (int): Number of decision variables.
            n_dimensions (int): Dimension of search space.
            n_iterations (int): Number of iterations.
            lower_bound (np.array): Lower bound array with the minimum possible values.
            upper_bound (np.array): Upper bound array with the maximum possible values.

        """

        logger.info('Creating class: Space')

        # Defining basic Space's variables
        self.n_agents = n_agents
        self.n_iterations = n_iterations
        self.n_variables = n_variables
        self.n_dimensions = n_dimensions

        # Space's agent-related variables
        self.agents = []
        self.best_agent = None

        # Space's bound-related variables
        self.lb = np.zeros(n_variables)
        self.ub = np.ones(n_variables)

        # Indicates whether the space is built or not
        self.built = False

        # Now, we need to build this class up
        self._build(lower_bound, upper_bound)

        # We will log some important information
        logger.info('Class created.')

    def __create_agents(self):
        """Creates and populates the agents array.
        Also defines a random best agent, only for initialization purposes.

        """

        logger.debug('Running private method: create_agents()')

        # Iterate through number of agents
        for i in range(self.n_agents):
            # Appends new agent to list
            self.agents.append(
                Agent(n_variables=self.n_variables, n_dimensions=self.n_dimensions))

        # Apply a random agent as the best one
        self.best_agent = self.agents[0]

        logger.debug('Agents created.')

    def __initialize_agents(self):
        """Initialize agents' position array with
        uniform random numbers.

        """

        logger.debug('Running private method: initialize_agents()')

        # Iterate through number of agents
        for agent in self.agents:
            # For every variable, we generate uniform random numbers
            for var in range(self.n_variables):
                agent.position[var] = r.generate_uniform_random_number(
                    self.lb[var], self.ub[var], size=self.n_dimensions)

        logger.debug('Agents initialized.')

    def __check_bound_size(self, bound, size):
        """Checks if the bounds' size are the same of
        variables size.

        Args:
            bound(np.array): bounds array.
            size(int): size to be checked.

        """

        logger.debug('Running private method: check_bound_size()')

        if len(bound) != size:
            e = f'Expected size is {size}. Got {len(bound)}.'
            logger.error(e)
            raise RuntimeError(e)
        else:
            logger.debug('Bound checked.')
            return True

    def _build(self, lower_bound, upper_bound):
        """This method will serve as the object building process.
        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            lower_bound (np.array): Lower bound array with the minimum possible values.
            upper_bound (np.array): Upper bound array with the maximum possible values.
        
        """

        logger.debug('Running private method: build()')

        # Checking if lower bound is avaliable and if its size
        # matches to our actual number of variables
        if lower_bound:
            if self.__check_bound_size(lower_bound, self.n_variables):
                self.lb = lower_bound
        else:
            e = f"Property 'lower_bound' cannot be {lower_bound}."
            logger.error(e)
            raise RuntimeError(e)

        # We need to check upper bounds as well
        if upper_bound:
            if self.__check_bound_size(upper_bound, self.n_variables):
                self.ub = upper_bound
        else:
            e = f"Property 'upper_bound' cannot be {upper_bound}."
            logger.error(e)
            raise RuntimeError(e)

        # As now we can assume the bounds are correct, we need to
        # create and initialize our agents
        self.__create_agents()
        self.__initialize_agents()

        # If no errors were shown, we can declared the Space as built
        self.built = True

        # Logging attributes
        logger.debug(
            f'Agents: {self.n_agents} | Iterations: {self.n_iterations}'
            + f' | Size: ({self.n_variables}, {self.n_dimensions}) | Lower Bound: {self.lb}'
            + f' | Upper Bound: {self.ub} | Built: {self.built}')

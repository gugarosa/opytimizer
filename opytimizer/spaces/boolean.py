import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.logging as l
from opytimizer.core.space import Space

logger = l.get_logger(__name__)

class BooleanSpace(Space):
    """A BooleanSpace class for agents, variables and methods
    related to the boolean search space.

    """
    def __init__(self, n_agents=1, n_variables=1, n_iterations=10):
        """Initialization method.

        Args:
            n_agents (int): Number of agents.
            n_variables (int): Number of decision variables.
            n_iterations (int): Number of iterations.

        """

        logger.info('Overriding class: Space -> BooleanSpace.')

        # Override its parent class with the receiving arguments
        super(BooleanSpace, self).__init__(n_agents, n_variables, n_iterations=n_iterations)

        # Defining the lower bound as an array of zeros
        lower_bound = np.zeros(n_variables)

        # Defining the upper bound as an array of ones
        upper_bound = np.ones(n_variables)

        # Now, we need to build this class up
        self._build(lower_bound, upper_bound)

        # Initializing agents
        self._initialize_agents()

        # We will log some important information
        logger.info('Class overrided.')

    def _initialize_agents(self):
        """Initialize agents' position array with boolean random numbers.

        """

        logger.debug('Running private method: initialize_agents().')

        # Iterate through all agents
        for agent in self.agents:
            # Iterate through all decision variables
            for j, (lb, ub) in enumerate(zip(self.lb, self.ub)):
                # For each decision variable, we generate uniform random numbers
                agent.position[j] = r.generate_binary_random_number(size=agent.n_dimensions)

                # Applies the lower bound the agent's lower bound
                agent.lb[j] = lb

                # And also the upper bound
                agent.ub[j] = ub

        logger.debug('Agents initialized.')

    def clip_limits(self):
        """Clips all agents' decision variables to the bounds limits.

        """

        # Iterates through all agents
        for agent in self.agents:
            # Iterates through all decision variables
            for j, (lb, ub) in enumerate(zip(self.lb, self.ub)):
                # Clips the array based on variables' lower and upper bounds
                agent.position[j] = np.clip(agent.position[j], lb, ub)

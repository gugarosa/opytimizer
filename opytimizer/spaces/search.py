import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.logging as l
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space

logger = l.get_logger(__name__)


class SearchSpace(Space):
    """A SearchSpace class for agents, variables and methods
    related to the search space.

    """

    def __init__(self, n_agents=1, n_variables=1, n_iterations=10, lower_bound=[0], upper_bound=[1]):
        """Initialization method.

        Args:
            n_agents (int): Number of agents.
            n_variables (int): Number of decision variables.
            n_iterations (int): Number of iterations.
            lower_bound (list): Lower bound list with the minimum possible values.
            upper_bound (list): Upper bound list with the maximum possible values.

        """

        logger.info('Overriding class: Space -> SearchSpace.')

        # Override its parent class with the receiving arguments
        super(SearchSpace, self).__init__(n_agents=n_agents,
                                          n_variables=n_variables, n_iterations=n_iterations)

        # Now, we need to build this class up
        self._build(lower_bound, upper_bound)

        # Initializing agents
        self._initialize_agents()

        # We will log some important information
        logger.info('Class overrided.')

    def _initialize_agents(self):
        """Initialize agents' position array with uniform random numbers.

        """

        logger.debug('Running private method: initialize_agents().')

        # Iterate through all agents
        for agent in self.agents:
            # Iterate through all decision variables
            for j, (lb, ub) in enumerate(zip(self.lb, self.ub)):
                # For each decision variable, we generate uniform random numbers
                agent.position[j] = r.generate_uniform_random_number(
                    lb, ub, size=agent.n_dimensions)

                # For each decision variable, we apply lower bound the agent's bound
                agent.lb[j] = lb

                # And also the upper bound
                agent.ub[j] = ub

        logger.debug('Agents initialized.')

    def check_limits(self):
        """Checks bounds limits of all agents and variables.

        """

        # Iterate through all agents
        for agent in self.agents:
            # Iterate through all decision variables
            for j, (lb, ub) in enumerate(zip(self.lb, self.ub)):
                # Clip the array based on variables' lower and upper bounds
                agent.position[j] = np.clip(agent.position[j], lb, ub)

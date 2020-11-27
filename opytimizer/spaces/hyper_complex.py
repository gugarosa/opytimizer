"""Hypercomplex-based search space.
"""

import opytimizer.math.random as r
import opytimizer.utils.logging as l
from opytimizer.core.space import Space

logger = l.get_logger(__name__)


class HyperComplexSpace(Space):
    """An HyperComplexSpace class that will hold agents, variables and methods
    related to the hypercomplex search space.

    """

    def __init__(self, n_agents=1, n_variables=1, n_dimensions=2, n_iterations=10,
                 lower_bound=(0,), upper_bound=(1,)):
        """Initialization method.

        Args:
            n_agents (int): Number of agents.
            n_variables (int): Number of decision variables.
            n_dimensions (int): Dimension of search space.
            n_iterations (int): Number of iterations.
            lower_bound (tuple): Lower bound tuple with the minimum possible values.
            upper_bound (tuple): Upper bound tuple with the maximum possible values.

        """

        logger.info('Overriding class: Space -> HyperComplexSpace.')

        # Override its parent class with the receiving arguments
        super(HyperComplexSpace, self).__init__(n_agents, n_variables, n_dimensions, n_iterations)

        # Now, we need to build this class up
        self._build(lower_bound, upper_bound)

        # Initializing agents
        self._initialize_agents()

        logger.info('Class overrided.')

    def _initialize_agents(self):
        """Initialize agents' position array with uniform random numbers.

        """

        logger.debug('Running private method: initialize_agents().')

        # Iterates through all agents
        for agent in self.agents:
            # Iterates through all decision variables
            for j in range(agent.n_variables):
                # For each decision variable, we generate uniform random numbers
                agent.position[j] = r.generate_uniform_random_number(size=agent.n_dimensions)

                # Applies the lower bound to the agent's lower bound
                agent.lb[j] = 0

                # And also the upper bound
                agent.ub[j] = 1

        logger.debug('Agents initialized.')

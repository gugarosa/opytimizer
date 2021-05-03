"""Water Cycle Algorithm.
"""

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class WCA(Optimizer):
    """A WCA class, inherited from Optimizer.

    This is the designed class to define WCA-related
    variables and methods.

    References:
        H. Eskandar.
        Water cycle algorithm – A novel metaheuristic optimization method for
        solving constrained engineering optimization problems.
        Computers & Structures (2012).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> WCA.')

        # Overrides its parent class with the receiving params
        super(WCA, self).__init__()

        # Number of rivers + sea
        self.nsr = 2

        # Maximum evaporation condition
        self.d_max = 0.1

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    @property
    def nsr(self):
        """float: Number of rivers summed with a single sea.

        """

        return self._nsr

    @nsr.setter
    def nsr(self, nsr):
        if not isinstance(nsr, int):
            raise e.TypeError('`nsr` should be an integer')
        if nsr < 1:
            raise e.ValueError('`nsr` should be > 1')

        self._nsr = nsr

    @property
    def d_max(self):
        """float: Maximum evaporation condition.

        """

        return self._d_max

    @d_max.setter
    def d_max(self, d_max):
        if not isinstance(d_max, (float, int)):
            raise e.TypeError('`d_max` should be a float or integer')
        if d_max < 0:
            raise e.ValueError('`d_max` should be >= 0')

        self._d_max = d_max

    @property
    def flows(self):
        """np.array: Array of flows.

        """

        return self._flows

    @flows.setter
    def flows(self, flows):
        if not isinstance(flows, np.ndarray):
            raise e.TypeError('`flows` should be a numpy array')

        self._flows = flows

    def create_additional_attrs(self, space):
        """Creates additional attributes that are used by this optimizer.

        Args:
            space (Space): A Space object containing meta-information.

        """

        # Our initial cost will be 0
        cost = 0

        # Array of flows
        self.flows = np.zeros(self.nsr, dtype=int)

        # For every river + sea
        for i in range(self.nsr):
            # We accumulates its fitness
            cost += space.agents[i].fit

        # Iterating again over rivers + sea
        for i in range(self.nsr):
            # Calculates its particular flow intensity (eq. 6)
            self.flows[i] = round(np.fabs(space.agents[i].fit / cost) * (len(space.agents) - self.nsr))

    def _raining_process(self, agents, best_agent):
        """Performs the raining process (eq. 12).

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.

        """

        # Iterate through every raindrop
        for k in range(self.nsr, len(agents)):
            # Calculates the euclidean distance between sea and raindrop / strream
            distance = (np.linalg.norm(best_agent.position - agents[k].position))

            # If distance if smaller than evaporation condition
            if distance > self.d_max:
                # Generates a new random gaussian number
                r1 = r.generate_gaussian_random_number(1, agents[k].n_variables)

                # Changes the stream position
                agents[k].position = best_agent.position + np.sqrt(0.1) * r1

    def _update_stream(self, agents):
        """Updates every stream position (eq. 8).

        Args:
            agents (list): List of agents.

        """

        # Defining a counter to the summation of flows
        n_flows = 0

        # For every river, ignoring the sea
        for k in range(1, self.nsr):
            # Accumulate the number of flows
            n_flows += self.flows[k]

            # Iterate through every possible flow
            for i in range((n_flows - self.flows[k]), n_flows):
                # Calculates a random uniform number between 0 and 1
                r1 = r.generate_uniform_random_number()

                # Updates stream position
                agents[i].position += r1 * 2 * (agents[i].position - agents[k].position)

    def _update_river(self, agents, best_agent):
        """Updates every river position (eq. 9).

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.

        """

        # For every river, ignoring the sea
        for k in range(1, self.nsr):
            # Calculates a random uniform number between 0 and 1
            r1 = r.generate_uniform_random_number()

            # Updates river position
            agents[k].position += r1 * 2 * (best_agent.position - agents[k].position)

    def update(self, space, n_iterations):
        """Wraps Water Cycle Algorithm over all agents and variables.

        Args:
            space (Space): Space containing agents and update-related information.
            n_iterations (int): Maximum number of iterations.

        """

        # Updates every stream position
        self._update_stream(space.agents)

        # Updates every river position
        self._update_river(space.agents, space.best_agent)

        # Sorts agents
        space.agents.sort(key=lambda x: x.fit)

        # Performs the raining process (eq. 12)
        self._raining_process(space.agents, space.best_agent)

        # Updates the evaporation condition
        self.d_max -= (self.d_max / n_iterations)

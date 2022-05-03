"""Water Cycle Algorithm.
"""

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


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

        logger.info("Overriding class: Optimizer -> WCA.")

        # Overrides its parent class with the receiving params
        super(WCA, self).__init__()

        # Number of sea + rivers
        self.nsr = 2

        # Maximum evaporation condition
        self.d_max = 0.1

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def nsr(self):
        """float: Number of rivers summed with a single sea."""

        return self._nsr

    @nsr.setter
    def nsr(self, nsr):
        if not isinstance(nsr, int):
            raise e.TypeError("`nsr` should be an integer")
        if nsr < 1:
            raise e.ValueError("`nsr` should be > 1")

        self._nsr = nsr

    @property
    def d_max(self):
        """float: Maximum evaporation condition."""

        return self._d_max

    @d_max.setter
    def d_max(self, d_max):
        if not isinstance(d_max, (float, int)):
            raise e.TypeError("`d_max` should be a float or integer")
        if d_max < 0:
            raise e.ValueError("`d_max` should be >= 0")

        self._d_max = d_max

    @property
    def flows(self):
        """np.array: Array of flows."""

        return self._flows

    @flows.setter
    def flows(self, flows):
        if not isinstance(flows, np.ndarray):
            raise e.TypeError("`flows` should be a numpy array")

        self._flows = flows

    def compile(self, space):
        """Compiles additional information that is used by this optimizer.

        Args:
            space (Space): A Space object containing meta-information.

        """

        # Array of flows
        self.flows = np.zeros(self.nsr, dtype=int)

    def _flow_intensity(self, agents):
        """Calculates the intensity of each possible flow (eq. 6).

        Args:
            agents (list): List of agents.

        """

        # Calculates the cost
        cost = np.sum([agents[i].fit for i in range(self.nsr)])

        # Iterates again over sea + rivers
        for i in range(self.nsr):
            # Calculates its particular flow intensity
            self.flows[i] = np.floor(
                np.fabs(agents[i].fit / cost) * (len(agents) - self.nsr)
            )

    def _raining_process(self, agents, best_agent):
        """Performs the raining process (eq. 11-12).

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.

        """

        # Iterates through all sea + rivers
        for i in range(0, self.nsr):
            # Iterates through all raindrops that belongs to specific sea or river
            for j in range(self.nsr, self.flows[i] + self.nsr):
                # Calculates the euclidean distance between sea and raindrop / stream
                distance = np.linalg.norm(best_agent.position - agents[j].position)

                # If distance if smaller than evaporation condition
                if distance < self.d_max:
                    # If it is supposed to replace the sea streams' position
                    if i == 0:
                        # Updates position (eq. 12)
                        r1 = r.generate_gaussian_random_number(1, agents[j].n_variables)
                        agents[j].position = best_agent.position + np.sqrt(0.1) * r1

                    # If it is supposed to replace the river streams' position
                    else:
                        # Updates position (eq. 11)
                        agents[j].fill_with_uniform()

    def _update_stream(self, agents, function):
        """Updates every stream position (eq. 8).

        Args:
            agents (list): List of agents.
            function (Function): A Function object that will be used as the objective function.

        """

        # Defines a counter to the summation of flows
        n_flows = 0

        # For every river, ignoring the sea
        for i in range(0, self.nsr):
            # Accumulates the number of flows
            n_flows += self.flows[i]

            # Iterates through every possible flow
            for j in range((self.nsr + n_flows - self.flows[i]), self.nsr + n_flows):
                # Calculates a random uniform number
                r1 = r.generate_uniform_random_number()

                # Updates river position
                agents[j].position += r1 * 2 * (agents[i].position - agents[j].position)

                # Clips its limits and recalculates its fitness
                agents[j].clip_by_bound()
                agents[j].fit = function(agents[j].position)

    def _update_river(self, agents, best_agent, function):
        """Updates every river position (eq. 9).

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.

        """

        # For every river, ignoring the sea
        for i in range(1, self.nsr):
            # Calculates a random uniform
            r1 = r.generate_uniform_random_number()

            # Updates river position
            agents[i].position += r1 * 2 * (best_agent.position - agents[i].position)

            # Clips its limits and recalculates its fitness
            agents[i].clip_by_bound()
            agents[i].fit = function(agents[i].position)

    def update(self, space, function, n_iterations):
        """Wraps Water Cycle Algorithm over all agents and variables.

        Args:
            space (Space): Space containing agents and update-related information.
            function (Function): A Function object that will be used as the objective function.
            n_iterations (int): Maximum number of iterations.

        """

        # Calculates the flow intensity
        self._flow_intensity(space.agents)

        # Updates every stream position
        self._update_stream(space.agents, function)

        # Updates every river position
        self._update_river(space.agents, space.best_agent, function)

        # Iterates through all rivers
        for i in range(1, self.nsr):
            # Iterates through all raindrops
            for j in range(self.nsr, len(space.agents)):
                # If raindrop position is better than river's
                if space.agents[j].fit < space.agents[i].fit:
                    # Exchanges their positions
                    space.agents[i], space.agents[j] = space.agents[j], space.agents[i]

        # Iterates through all rivers:
        for i in range(1, self.nsr):
            # If river position is better than seá's
            if space.agents[i].fit < space.agents[0].fit:
                # Exchanges their positions
                space.agents[i], space.agents[0] = space.agents[0], space.agents[i]

        # Performs the raining process (eq. 12)
        self._raining_process(space.agents, space.best_agent)

        # Updates the evaporation condition
        self.d_max -= self.d_max / n_iterations

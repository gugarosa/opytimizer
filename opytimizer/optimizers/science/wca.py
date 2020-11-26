"""Water Cycle Algorithm.
"""

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
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

    def __init__(self, algorithm='WCA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> WCA.')

        # Override its parent class with the receiving hyperparams
        super(WCA, self).__init__(algorithm)

        # Number of rivers + sea
        self.nsr = 2

        # Maximum evaporation condition
        self.d_max = 0.1

        # Now, we need to build this class up
        self._build(hyperparams)

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

    def _flow_intensity(self, agents):
        """Calculates the intensity of each possible flow (eq. 6).

        Args:
            agents (list): List of agents.

        Returns:
            It returns an array of flows' intensity.

        """

        # Our initial cost will be 0
        cost = 0

        # Creates an empty integer array of number of rivers + sea
        flows = np.zeros(self.nsr, dtype=int)

        # For every river + sea
        for i in range(self.nsr):
            # We accumulates its fitness
            cost += agents[i].fit

        # Iterating again over rivers + sea
        for i in range(self.nsr):
            # Calculates its particular flow intensity
            flows[i] = round(np.fabs(agents[i].fit / cost) * (len(agents) - self.nsr))

        return flows

    def _raining_process(self, agents, best_agent):
        """Performs the raining process (eq. 12).

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.

        """

        # Iterate through every raindrop
        for k in range(self.nsr, len(agents)):
            # Calculate the euclidean distance between sea and raindrop / strream
            distance = (np.linalg.norm(best_agent.position - agents[k].position))

            # If distance if smaller than evaporation condition
            if distance > self.d_max:
                # Generates a new random gaussian number
                r1 = r.generate_gaussian_random_number(1, agents[k].n_variables)

                # Changes the stream position
                agents[k].position = best_agent.position + np.sqrt(0.1) * r1

    def _update_stream(self, agents, flows):
        """Updates every stream position (eq. 8).

        Args:
            agents (list): List of agents.
            flows (np.array): Array of flows' intensity.

        """

        # Defining a counter to the summation of flows
        n_flows = 0

        # For every river, ignoring the sea
        for k in range(1, self.nsr):
            # Accumulate the number of flows
            n_flows += flows[k]

            # Iterate through every possible flow
            for i in range((n_flows - flows[k]), n_flows):
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

    def _update(self, agents, best_agent, flows):
        """Updates the agents position.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            flows (np.array): Array of flows' intensity.

        """

        # Updates every stream position
        self._update_stream(agents, flows)

        # Updates every river position
        self._update_river(agents, best_agent)

    def run(self, space, function, store_best_only=False, pre_evaluation=None):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (bool): If True, only the best agent of each iteration is stored in History.
            pre_evaluation (callable): This function is executed before evaluating the function being optimized.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Initial search space evaluation
        self._evaluate(space, function, hook=pre_evaluation)

        # Calculating the flow's intensity (eq. 6)
        flows = self._flow_intensity(space.agents)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.file(f'Iteration {t+1}/{space.n_iterations}')

                # Updating agents
                self._update(space.agents, space.best_agent, flows)

                # Checking if agents meet the bounds limits
                space.clip_limits()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, hook=pre_evaluation)

                # Sorting agents
                space.agents.sort(key=lambda x: x.fit)

                # Performs the raining process (eq. 12)
                self._raining_process(space.agents, space.best_agent)

                # Updates the evaporation condition
                self.d_max -= (self.d_max / space.n_iterations)

                # Every iteration, we need to dump agents and best agent
                history.dump(agents=space.agents, best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.file(f'Fitness: {space.best_agent.fit}')
                logger.file(f'Position: {space.best_agent.position}')

        return history

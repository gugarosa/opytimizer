import copy

import numpy as np
import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class WCA(Optimizer):
    """A WCA class, inherited from Optimizer.

    This will be the designed class to define WCA-related
    variables and methods.

    References:


    """

    def __init__(self, algorithm='WCA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): A string holding optimizer's algorithm name.
            hyperparams (dict): An hyperparams dictionary containing key-value
                parameters to meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> WCA.')

        # Override its parent class with the receiving hyperparams
        super(WCA, self).__init__(algorithm=algorithm)

        # Number of rivers + sea
        self._nsr = 2

        # Maximum evaporation condition
        self._d_max = 0.1

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
        self._nsr = nsr

    @property
    def d_max(self):
        """float: Maximum evaporation condition.

        """

        return self._d_max

    @d_max.setter
    def d_max(self, d_max):
        self._d_max = d_max

    def _build(self, hyperparams):
        """This method will serve as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            hyperparams (dict): An hyperparams dictionary containing key-value
                parameters to meta-heuristics.

        """

        logger.debug('Running private method: build().')

        # We need to save the hyperparams object for faster looking up
        self.hyperparams = hyperparams

        # If one can find any hyperparam inside its object,
        # set them as the ones that will be used
        if hyperparams:
            if 'nsr' in hyperparams:
                self.nsr = hyperparams['nsr']
            if 'd_max' in hyperparams:
                self.d_max = hyperparams['d_max']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: nsr = {self.nsr}, d_max = {self.d_max} | Built: {self.built}.')

    def _flow_intensity(self, agents):
        """Calculates the intensity of each possible flow.

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
            flows[i] = round(np.fabs(agents[i].fit / cost)
                             * (len(agents) - self.nsr))

        return flows

    def _raining_process(self, agents, flows):
        """Performs the raining process.

        Args:
            agents (list): List of agents.
            flows (np.array): Array of flows' intensity.

        """

        #

        return

    def _update_stream(self, agents, best_agent, flows):
        """Updates every stream position.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            flows (np.array): Array of flows' intensity.

        """

        # Defining a counter to the summation of flows
        n_flows = 0

        # For every river, ignoring the
        for k in range(1, self.nsr):
            # Accumulate the number of flows
            n_flows += flows[k]

            # Iterate through every possible flow
            for i in range((n_flows - flows[k]), n_flows):
                # Calculates a random uniform number between 0 and 1
                r1 = r.generate_uniform_random_number()

                # Updates stream position
                agents[i].position += r1 * 2 * \
                    (agents[i].position - agents[k].position)

    def _update_river(self, agents, best_agent):
        """Updates every river position.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.

        """

        # For every river, ignoring the sea
        for k in range(1, self.nsr):
            # Calculates a random uniform number between 0 and 1
            r1 = r.generate_uniform_random_number()

            # Updates river position
            agents[k].position += r1 * 2 * \
                (best_agent.position - agents[k].position)

    def _update(self, agents, best_agent, flows):
        """Updates the agents position.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            flows (np.array): Array of flows' intensity.

        """

        # Updates every stream position (Equation 8)
        self._update_stream(agents, best_agent, flows)

        # Updates every river position (Equation 9)
        self._update_river(agents, best_agent)

    def run(self, space, function):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Initial search space evaluation
        self._evaluate(space, function)

        # Calculating the flow's intensity (Equation 6)
        flows = self._flow_intensity(space.agents)

        # We will define a History object for further dumping
        history = h.History()

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents, space.best_agent, flows)

            # Checking if agents meets the bounds limits
            space.check_bound_limits(space.agents, space.lb, space.ub)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            # Sorting agents
            space.agents.sort(key=lambda x: x.fit)

            # Performs the raining process
            self._raining_process(space.agents, flows)

            # Updates the evaporation condition
            self.d_max -= (self.d_max / space.n_iterations)

            # Every iteration, we need to dump the current space agents
            history.dump(space.agents, space.best_agent)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

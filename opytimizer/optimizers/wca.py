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

    def _flow_intensity(self):
        """
        """

        #


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

        # Calculating the flow's intensity
        self._flow_intensity()

        # We will define a History object for further dumping
        history = h.History()

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents, space.best_agent, function, t,
                         self.frequency, self.velocity, self.loudness, self.pulse_rate)

            # Checking if agents meets the bounds limits
            space.check_bound_limits(space.agents, space.lb, space.ub)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            # Every iteration, we need to dump the current space agents
            history.dump(space.agents, space.best_agent)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

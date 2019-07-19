import numpy as np

import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.optimizers.hs import HS

logger = l.get_logger(__name__)


class IHS(HS):
    """An IHS class, inherited from HS.

    This will be the designed class to define IHS-related
    variables and methods.

    References:
        M. Mahdavi, M. Fesanghary, and E. Damangir. An improved harmony search algorithm for solving optimization problems. Applied Mathematics and Computation (2007). 

    """

    def __init__(self, algorithm='IHS', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): A string holding optimizer's algorithm name.
            hyperparams (dict): An hyperparams dictionary containing key-value
                parameters to meta-heuristics.

        """

        logger.info('Overriding class: HS -> IHS.')

        # Override its parent class with the receiving hyperparams
        super(IHS, self).__init__(
            algorithm=algorithm, hyperparams=hyperparams)

        # Minimum pitch adjusting rate
        self._PAR_min = 0

        # Maximum pitch adjusting rate
        self._PAR_max = 1

        # Minimum bandwidth parameter
        self._bw_min = 1

        # Maximum bandwidth parameter
        self._bw_max = 10

        # Now, we need to re-build this class up
        self._rebuild()

        logger.info('Class overrided.')

    @property
    def PAR_min(self):
        """float: Minimum pitch adjusting rate.

        """

        return self._PAR_min

    @PAR_min.setter
    def PAR_min(self, PAR_min):
        self._PAR_min = PAR_min

    @property
    def PAR_max(self):
        """float: Maximum pitch adjusting rate.

        """

        return self._PAR_max

    @PAR_max.setter
    def PAR_max(self, PAR_max):
        self._PAR_max = PAR_max

    @property
    def bw_min(self):
        """float: Minimum bandwidth parameter.

        """

        return self._bw_min

    @bw_min.setter
    def bw_min(self, bw_min):
        self._bw_min = bw_min

    @property
    def bw_max(self):
        """float: Maximum bandwidth parameter.

        """

        return self._bw_max

    @bw_max.setter
    def bw_max(self, bw_max):
        self._bw_max = bw_max

    def _rebuild(self):
        """This method will serve as the object re-building process.

        One is supposed to use this class only when defining extra hyperparameters
        that can not be inherited by its parent.

        """

        logger.debug('Running private method: rebuild().')

        # If one can find any hyperparam inside its object,
        # set them as the ones that will be used
        if self.hyperparams:
            if 'PAR_min' in self.hyperparams:
                self.PAR_min = self.hyperparams['PAR_min']
            if 'PAR_max' in self.hyperparams:
                self.PAR_max = self.hyperparams['PAR_max']
            if 'bw_min' in self.hyperparams:
                self.bw_min = self.hyperparams['bw_min']
            if 'bw_max' in self.hyperparams:
                self.bw_max = self.hyperparams['bw_max']

        # Logging attributes
        logger.debug(
            f'Additional hyperparameters: PAR_min = {self.PAR_min}, PAR_max = {self.PAR_max}, bw_min = {self.bw_min}, bw_max = {self.bw_max}.')

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

        # We will define a History object for further dumping
        history = h.History()

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating pitch adjusting rate
            self.PAR = self.PAR_min + \
                (((self.PAR_max - self.PAR_min) / space.n_iterations) * t)

            # Updating bandwidth parameter
            self.bw = self.bw_max * \
                np.exp((np.log(self.bw_min / self.bw_max) / space.n_iterations) * t)

            # Updating agents
            self._update(space.agents, space.lb, space.ub, function)

            # Checking if agents meets the bounds limits
            space.check_limits(space.agents, space.lb, space.ub)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            # Every iteration, we need to dump the current space agents
            history.dump(space.agents, space.best_agent)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

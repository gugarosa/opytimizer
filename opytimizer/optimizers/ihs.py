import numpy as np

import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.optimizers.hs import HS

logger = l.get_logger(__name__)


class IHS(HS):
    """An IHS class, inherited from HS.

    This is the designed class to define IHS-related
    variables and methods.

    References:
        M. Mahdavi, M. Fesanghary, and E. Damangir. An improved harmony search algorithm for solving optimization problems. Applied Mathematics and Computation (2007). 

    """

    def __init__(self, algorithm='IHS', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: HS -> IHS.')

        # Override its parent class with the receiving hyperparams
        super(IHS, self).__init__(
            algorithm=algorithm, hyperparams=hyperparams)

        # Minimum pitch adjusting rate
        self.PAR_min = 0

        # Maximum pitch adjusting rate
        self.PAR_max = 1

        # Minimum bandwidth parameter
        self.bw_min = 1

        # Maximum bandwidth parameter
        self.bw_max = 10

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
        if not (isinstance(PAR_min, float) or isinstance(PAR_min, int)):
            raise e.TypeError('`PAR_min` should be a float or integer')
        if PAR_min < 0 or PAR_min > 1:
            raise e.ValueError('`PAR_min` should be between 0 and 1')

        self._PAR_min = PAR_min

    @property
    def PAR_max(self):
        """float: Maximum pitch adjusting rate.

        """

        return self._PAR_max

    @PAR_max.setter
    def PAR_max(self, PAR_max):
        if not (isinstance(PAR_max, float) or isinstance(PAR_max, int)):
            raise e.TypeError('`PAR_max` should be a float or integer')
        if PAR_max < 0 or PAR_max > 1:
            raise e.ValueError('`PAR_max` should be between 0 and 1')
        if PAR_max < self.PAR_min:
            raise e.ValueError('`PAR_max` should be >= `PAR_min`')

        self._PAR_max = PAR_max

    @property
    def bw_min(self):
        """float: Minimum bandwidth parameter.

        """

        return self._bw_min

    @bw_min.setter
    def bw_min(self, bw_min):
        if not (isinstance(bw_min, float) or isinstance(bw_min, int)):
            raise e.TypeError('`bw_min` should be a float or integer')
        if bw_min < 0:
            raise e.ValueError('`bw_min` should be >= 0')

        self._bw_min = bw_min

    @property
    def bw_max(self):
        """float: Maximum bandwidth parameter.

        """

        return self._bw_max

    @bw_max.setter
    def bw_max(self, bw_max):
        if not (isinstance(bw_max, float) or isinstance(bw_max, int)):
            raise e.TypeError('`bw_max` should be a float or integer')
        if bw_max < 0:
            raise e.ValueError('`bw_max` should be >= 0')
        if bw_max < self.bw_min:
            raise e.ValueError('`bw_max` should be >= `bw_min`')

        self._bw_max = bw_max

    def _rebuild(self):
        """This method serves as the object re-building process.

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

    def run(self, space, function, store_best_only=False, pre_evaluation_hook=None):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (boolean): If True, only the best agent of each iteration is stored in History.
            pre_evaluation_hook (function): A function that receives the optimizer, space and function
                and returns None. This function is executed before evaluating the function being optimized.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Check if there is a pre-evaluation hook
        if pre_evaluation_hook:
            # Applies the hook
            pre_evaluation_hook(self, space, function)

        # Initial search space evaluation
        self._evaluate(space, function)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

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
            self._update(space.agents, function)

            # Checking if agents meets the bounds limits
            space.check_limits()

            # Check if there is a pre-evaluation hook
            if pre_evaluation_hook:
                # Applies the hook
                pre_evaluation_hook(self, space, function)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            # Every iteration, we need to dump agents and best agent
            history.dump(agents=space.agents, best_agent=space.best_agent)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

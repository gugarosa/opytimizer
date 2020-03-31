import time

import opytimizer.utils.exception as e
import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Opytimizer:
    """An Opytimizer class holds all the information needed
    in order to perform an optimization task.

    """

    def __init__(self, space=None, optimizer=None, function=None):
        """Initialization method.

        Args:
            space (Space): A Space's object.
            optimizer (Optimizer): An Optimizer's object, where it can be a child (e.g., PSO, BA, etc).
            function (Function): A Function's object, where it can be a child (e.g., MultiFunction).

        """

        logger.info('Creating class: Opytimizer.')

        # Attaches the space to Opytimizer
        self.space = space

        # Attaches the optimizer
        self.optimizer = optimizer

        # Lastly, attaches the function
        self.function = function

        # We will log some important information
        logger.debug(f'Space: {self.space} | Optimizer: {self.optimizer} | Function: {self.function}.')

        logger.info('Class created.')

    @property
    def space(self):
        """Space: A Space's object.
        """

        return self._space

    @space.setter
    def space(self, space):
        if not space.built:
            raise e.BuildError(
                '`space` should be built before using Opytimizer')

        self._space = space

    @property
    def optimizer(self):
        """Optimizer: An Optimizer's object, where it can be a child (PSO, BA, etc).
        """

        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if not optimizer.built:
            raise e.BuildError(
                '`optimizer` should be built before using Opytimizer')

        self._optimizer = optimizer

    @property
    def function(self):
        """Function: A Function's object, where it can be a child (External or Internal).
        """

        return self._function

    @function.setter
    def function(self, function):
        if not function.built:
            raise e.BuildError(
                '`function` should be built before using Opytimizer')

        self._function = function

    def start(self, store_best_only=False, pre_evaluation_hook=None):
        """Starts the optimization task.

        Args
            store_best_only (bool): If True, only the best agent of each iteration is stored in History.
            pre_evaluation_hook (callable): This function is executed before evaluating the function being optimized.

        Returns:
            A History object describing the agents position and best fitness values
                at each iteration throughout the optimization process.

        """

        logger.info('Starting optimization task.')

        # Starting timer to count optimization task
        start = time.time()

        # Starting optimizer
        history = self.optimizer.run(self.space, self.function, store_best_only, pre_evaluation_hook)

        # Ending timer
        end = time.time()

        # Calculating optimization task time
        opt_time = end - start

        # Dumping the elapsed time to optimization history
        history.dump(time=opt_time)

        logger.info('Optimization task ended.')
        logger.info(f'It took {opt_time} seconds.')

        return history

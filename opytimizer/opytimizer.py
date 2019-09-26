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
        logger.debug(
            f'Space: {self.space} | Optimizer: {self.optimizer} | Function: {self.function}.')

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

    def start(self):
        """Starts the optimization task.

        Returns:
            A History object describing the agents position and best fitness values
                at each iteration throughout the optimization process.

        """

        logger.info('Starting optimization task.')

        # Starting timer to count optimization task
        start = time.time()

        # Starting optimizer
        opt_history = self.optimizer.run(self.space, self.function)

        # Ending timer
        end = time.time()

        logger.info('Optimization task ended.')
        logger.info(f'It took {(end - start)} seconds.')

        return opt_history

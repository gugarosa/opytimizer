import time

import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Opytimizer:
    """An Opytimizer class that will hold all the information needed
    in order to perform an optimization task.

    Properties:
        space (Space): A Space's object.
        optimizer (Optimizer): An Optimizer's object, where it can be a child (PSO, BA, etc).
        function (Function): A Function's object, where it can be a child (External or Internal).

    Methods:
        _is_built(entity): Checks whether a miscellaneous entity is built or not.
        start(): Starts the optimization task.

    """

    def __init__(self, space=None, optimizer=None, function=None):
        """Initialization method.

        Args:
            space (Space): A Space's object.
            optimizer (Optimizer): An Optimizer's object, where it can be a child (PSO, BA, etc).
            function (Function): A Function's object, where it can be a child (External or Internal).

        """

        logger.info('Creating class: Opytimizer.')

        # Checks if Space is built
        if self._is_built(space):
            self.space = space

        # Checks if Optimizer is built
        if self._is_built(optimizer):
            self.optimizer = optimizer

        # Checks if Function is built
        if self._is_built(function):
            self.function = function

        # We will log some important information
        logger.debug(
            f'Space: {self.space} | Optimizer: {self.optimizer} | Function: {self.function}')

        logger.info('Class created.')

    @property
    def space(self):
        """A Space's object.
        """

        return self._space

    @space.setter
    def space(self, space):
        self._space = space

    @property
    def optimizer(self):
        """An Optimizer's object, where it can be a child (PSO, BA, etc).
        """

        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def function(self):
        """A Function's object, where it can be a child (External or Internal).
        """

        return self._function

    @function.setter
    def function(self, function):
        self._function = function

    def _is_built(self, entity):
        """Checks whether a miscellaneous entity is built or not.

        Args:
            entity (obj): A miscellaneous entity that has the built attribute.

        Returns:
            True, if entity is built.

        """

        if entity.built:
            return True
        else:
            e = entity.__class__.__name__ + 'is not built yet.'
            logger.error(e)
            raise RuntimeError(e)

    def start(self):
        """Starts the optimization task.

        """

        logger.info('Starting optimization task.')

        # Starting timer to count optimization task
        start = time.time()

        # Starting optimizer
        self.optimizer.run(self.space, self.function)
        
        # Ending timer, still needs to get the diff % 60 for real seconds
        end = time.time()

        logger.info('Optimization task ended.')
        logger.info(f'It took {(end - start) % 60} seconds.')

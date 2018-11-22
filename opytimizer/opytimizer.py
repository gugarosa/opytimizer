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

    """

    def __init__(self, space=None, optimizer=None, function=None):
        """Initialization method.

        Args:
            space (Space): A Space's object.
            optimizer (Optimizer): An Optimizer's object, where it can be a child (PSO, BA, etc).
            function (Function): A Function's object, where it can be a child (External or Internal).

        """

        logger.info('Creating class: Opytimizer')

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

    def _is_built(self, entity):
        """Checks whether a miscellaneous entity is built or not.

        Args:
            entity (obj): A miscellaneous entity that has the built attribute.

        """

        if entity.built:
            return True
        else:
            e = entity.__class__.__name__ + 'is not built yet.'
            logger.error(e)
            raise RuntimeError(e)

    def start(self):
        """
        """
        self.optimizer.run(self.space, self.function)
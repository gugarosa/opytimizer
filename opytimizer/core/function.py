"""Objective function.
"""

from inspect import signature

import opytimizer.utils.exception as e
import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Function:
    """A Function class that holds a single objective function.

    """

    def __init__(self, pointer=callable):
        """Initialization method.

        Args:
            pointer (callable): Pointer to a function that will return the fitness value.

        """

        logger.info('Creating class: Function.')

        # Pointer
        self.pointer = pointer

        # Pointer's name
        if hasattr(pointer, '__name__'):
            self.name = pointer.__name__
        else:
            self.name = pointer.__class__.__name__

        # If no errors were shown, we can declare the function as built
        self.built = True

        # Logs the attributes
        logger.debug('Function: %s | Built: %s.',
                     self.name, self.built)
        logger.info('Class created.')

    def __call__(self, x):
        """Callable to avoid using the `pointer` property.

        Args:
            x (np.array): Array of positions.

        Returns:
            Objective function fitness.

        """

        return self.pointer(x)

    @property
    def pointer(self):
        """callable: Points to the actual function.

        """

        return self._pointer

    @pointer.setter
    def pointer(self, pointer):
        if not callable(pointer):
            raise e.TypeError('`pointer` should be a callable')
        if len(signature(pointer).parameters) > 1:
            raise e.ArgumentError('`pointer` should only have 1 argument')

        self._pointer = pointer

    @property
    def name(self):
        """str: Name of the function.

        """

        return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise e.TypeError('`name` should be a string')

        self._name = name

    @property
    def built(self):
        """bool: Indicates whether the function is built.

        """

        return self._built

    @built.setter
    def built(self, built):
        self._built = built

from inspect import signature

import opytimizer.utils.exception as e
import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Function:
    """A Function class for using with objective functions
    that will be further evaluated.

    It serves as the basis class for holding in-code related
    objective functions.

    """

    def __init__(self, pointer=callable):
        """Initialization method.

        Args:
            function (callable): This should be a pointer to a function
                that will return the fitness value.

        """

        logger.info('Creating class: Function.')

        # Also, we need a callable to point to the actual function
        self.pointer = pointer

        # Indicates whether the function is built or not
        self.built = False

        # Now, we need to build this class up
        self._build(pointer)

        logger.info('Class created.')

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
    def built(self):
        """bool: Indicate whether the function is built.

        """

        return self._built

    @built.setter
    def built(self, built):
        self._built = built

    def _build(self, pointer):
        """This method will serve as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            function (callable): This should be a pointer to a function
                that will return the fitness value.

        """

        logger.debug('Running private method: build().')

        # We apply to class pointer's the desired function
        self.pointer = pointer

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(f'Pointer: {self.pointer.__name__} | Built: {self.built}')

import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Function:
    """A Function class to hold objective functions
        that will be further evaluated.

    It will server as the basis class for holding in-code related
    objective functions.

    Attributes:
        pointer (*func): This should be a pointer to a function that will
            return the fitness value.
        built (boolean): A boolean to indicate whether the function is built.

    Methods:
        _build(function): Sets an external function point
            to a class attribute.

    """

    def __init__(self, pointer=callable):
        """Initialization method.

        Args:
            function (*func): This should be a pointer to a function
                that will return the fitness value.

        """

        logger.info('Creating class: Function.')

        # Also, we need a pointer to point to our actual function
        self._pointer = callable

        # Indicates whether the function is built or not
        self._built = False

        # Now, we need to build this class up
        self._build(pointer)

        logger.info('Class created.')

    @property
    def pointer(self):
        """A pointer to point to our actual function.

        """

        return self._pointer

    @pointer.setter
    def pointer(self, pointer):
        self._pointer = pointer

    @property
    def built(self):
        """A boolean to indicate whether the function is built.

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
            function (*func): This should be a pointer to a function
                that will return the fitness value.

        """

        logger.debug('Running private method: build().')

        # We apply to class pointer's the desired function
        if pointer:
            self.pointer = pointer
        else:
            e = f"Property 'pointer' cannot be {pointer}."
            logger.error(e)
            raise RuntimeError(e)

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Pointer: {self.pointer} | Built: {self.built}')

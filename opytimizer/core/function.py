import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Function:
    """A Function class to hold objective functions that
    will be further evaluated.

    Properties:
        function_type (str): Type of function (internal or external).
        pointer (*func): This should be a pointer to a function that will
        return the fitness value.
        built (boolean): A boolean to indicate whether the function is built.

    """

    def __init__(self, function_type='internal'):
        """Initialization method.

        Args:
            type (str): Type of function (internal or external).

        """

        # We define the functions's type
        self._type = function_type

        # Also, we need a pointer to point to our actual function
        self._pointer = None

        # Indicates whether the function is built or not
        self._built = False

    @property
    def type(self):
        """Type of function (internal or external).
        """

        return self._type

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

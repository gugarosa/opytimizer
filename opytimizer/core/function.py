import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Function:
    """A Function class to hold objective functions that
    will be further evaluated.

    Properties:
        built (bool): A boolean to indicate whether the function is built.
        pointer (*func): This should be a pointer to a function that will
        return the fitness value.
        type (str): Type of function (internal or external).

    """

    def __init__(self, type='internal'):
        """Initialization method.

        Args:
            type (str): Type of function (internal or external).

        """

        # We define the functions's type
        self.type = type

        # Also, we need a pointer to point to our actual function
        self.pointer = None

        # Indicates whether the function is built or not
        self.built = False

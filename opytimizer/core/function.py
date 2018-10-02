import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Function:
    """A Function class to hold objective functions that
    will be further evaluated.

    Properties:
        type (str): Type of function (internal or external).
        call (*func): This should be a pointer to a function that will
        return the fitness value.
        _built (bool): A boolean to indicate whether the function is built.

    """

    def __init__(self, type='internal'):
        """Initialization method.

        Args:
            type (str): Type of function (internal or external).

        """

        logger.info('Initializing class: Function')

        # Apply arguments to internal variables
        self.type = type

        # Variables that can be accessed from outside
        self.call = None

        # Internal use only
        self._built = False

        # We will log some important information
        logger.info('Function created.')

import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Function:
    """
    """

    def __init__(self, type='internal'):
        """
        """

        logger.info('Initializing class: Function')

        # Apply arguments to internal variables
        self.type = type

        # Variables that can be accessed from outside
        self.function = None

        # Internal use only
        self._built = False

        # We will log some important information
        logger.info('Function created.')
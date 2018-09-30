import opytimizer.functions.internal as internal
import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Function:
    """
    """

    def __init__(self, type='internal'):
        """
        """

        logger.info('Initializing Function ...')

        # Apply arguments to internal variables
        self.type = type

        # Variables that can be accessed from outside
        self.function = None

        # Internal use only
        self._built = False

        # We will log some important information
        logger.info('Function created.')

    def build(self, expression):
        """
        """

        logger.debug('Running method: build()')

        if self.type == 'internal':
            # Internal functions
            self.function = internal.build_internal(expression=expression)

        # Set internal built variable to 'True'
        self._built = True

import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Optimizer:
    """
    """

    def __init__(self, hyperparams=None):
        """
        """

        logger.info('Initializing class: Optimizer')

        # Apply arguments to internal variables
        self.hyperparams = hyperparams

        # Variables that can be accessed from outside
        self.algorithm = None

        # Internal use only
        self._built = False

        # We will log some important information
        logger.info('Optimizer created.')

    def build(self):
        """
        """

        logger.debug('Running method: build()')

        self._built = True

        logger.debug('Optimizer was successfully built.')
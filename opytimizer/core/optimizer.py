import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Optimizer:
    """An Optimizer class that will serve as meta-heuristic
    techniques' parent.

    Properties:
        hyperparams (dict): An hyperparams dictionary containing key-value
        parameters to meta-heuristics.
        algorithm (str): A string indicating the algorithm name.
        _built (bool): A boolean to indicate whether the function is built.

    Methods:
        build(): An object building method.
        call(): This will be overrided by an Optimizer child. 

    """

    def __init__(self, hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): An hyperparams dictionary containing key-value
            parameters to meta-heuristics.

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
        """This method will serve as the object building process.
        One can define several functions here that does not necessarily
        needs to be on its initialization.

        """

        logger.debug('Running method: build()')

        self._built = True

        logger.debug('Optimizer was successfully built.')

    def call(self):
        """A call method that will be overrided by an Optimizer
        child. 

        """

        pass

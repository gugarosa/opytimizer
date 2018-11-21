import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Optimizer:
    """An Optimizer class that will serve as meta-heuristic
    techniques' parent.

    Properties:
        algorithm (str): A string indicating the algorithm name.
        built (bool): A boolean to indicate whether the optimizer is built.
        hyperparams (dict): An hyperparams dictionary containing key-value
        parameters to meta-heuristics.
    """

    def __init__(self, algorithm='PSO'):
        """Initialization method.

        Args:
            algorithm (str): A string indicating the algorithm name.

        """

        # We define the algorithm's name
        self.algorithm = algorithm

        # Also, we need a dict of desired hyperparameters
        self.hyperparams = None

        # Indicates whether the optimizer is built or not
        self.built = False

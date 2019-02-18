import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Optimizer:
    """An Optimizer class that will serve as meta-heuristic
    techniques' parent.

    Properties:
        algorithm (str): A string indicating the algorithm name.
        hyperparams (dict): An hyperparams dictionary containing key-value
        parameters to meta-heuristics.
        built (boolean): A boolean to indicate whether the optimizer is built.

    Methods:
        _update(agents): Updates the agents' position array.
        _evaluate(space, function): Evaluates the search space according to the objective function.
        run(space, function): Runs the optimization pipeline.

    """

    def __init__(self, algorithm='PSO'):
        """Initialization method.

        Args:
            algorithm (str): A string indicating the algorithm name.

        """

        # We define the algorithm's name
        self._algorithm = algorithm

        # Also, we need a dict of desired hyperparameters
        self._hyperparams = None

        # Indicates whether the optimizer is built or not
        self._built = False

    @property
    def algorithm(self):
        """A string indicating the algorithm name.
        """

        return self._algorithm

    @property
    def hyperparams(self):
        """A dictionary containing key-value parameters to meta-heuristics.
        """

        return self._hyperparams

    @property
    def built(self):
        """A boolean to indicate whether the optimizer is built.
        """

        return self._built

    def _update(self, agents):
        """Updates the agents' position array.
        As each optimizer child can have a different
        procedure of update, you will need to implement
        it directly on child's class.

        Args:
            agents (list): A list of agents that will be updated.

        """

        raise NotImplementedError

    def _evaluate(self, space, function):
        """Evaluates the search space according to the objective function.
        As each optimizer child can have a different
        procedure of evaluation, you will need to implement
        it directly on child's class.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        """

        raise NotImplementedError

    def run(self, space, function):
        """Runs the optimization pipeline.
        As each optimizer child can have a different
        pipeline, you will need to implement it directly
        on child's class.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        """

        raise NotImplementedError

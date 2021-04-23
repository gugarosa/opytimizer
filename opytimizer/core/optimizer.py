"""Optimizer.
"""

import copy

import opytimizer.utils.decorator as d
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Optimizer:
    """An Optimizer class that serves as meta-heuristics' parent.

    """

    def __init__(self, algorithm=''):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm's name.

        """

        # Algorithm's name
        self.algorithm = algorithm

        # Key-value hyperparameters
        self.params = None

        # Indicates whether the optimizer is built or not
        self.built = False

    @property
    def algorithm(self):
        """str: Algorithm's name.

        """

        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm):
        self._algorithm = algorithm

    @property
    def params(self):
        """dict: Key-value hyperparameters.

        """

        return self._params

    @params.setter
    def params(self, params):
        if not (isinstance(params, dict) or params is None):
            raise e.TypeError('`params` should be a dictionary')

        self._params = params

    @property
    def built(self):
        """bool: Indicates whether the optimizer is built.

        """

        return self._built

    @built.setter
    def built(self, built):
        self._built = built

    def _build(self, params):
        """This method serves as the object building process.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.debug('Running private method: build().')

        # We need to save the params object for faster looking up
        self.params = params

        # Checks if params are really provided
        if params:
            # If one can find any hyperparam inside its object
            for k, v in params.items():
                # Set it as the one that will be used
                setattr(self, k, v)

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug('Algorithm: %s | Hyperparameters: %s | '
                     'Built: %s.',
                     self.algorithm, str(params),
                     self.built)

    def update(self):
        """Updates the agents' position array.

        As each optimizer child can have a different procedure of update,
        you will need to implement it directly on child's class.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError

    @d.pre_evaluate
    def evaluate(self, space, function):
        """Evaluates the search space according to the objective function.

        If you need a specific evaluate method, please re-implement it on child's class.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object serving as an objective function.

        """

        # Iterates through all agents
        for agent in space.agents:
            # Calculates the fitness value of current agent
            agent.fit = function(agent.position)

            # If agent's fitness is better than global fitness
            if agent.fit < space.best_agent.fit:
                # Makes a deep copy of agent's position and fitness
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)

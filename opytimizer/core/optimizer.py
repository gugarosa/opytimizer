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
            algorithm (str): Indicates the algorithm name.

        """

        # We define the algorithm's name
        self.algorithm = algorithm

        # Also, we need a dict of desired hyperparameters
        self.hyperparams = {}

        # Indicates whether the optimizer is built or not
        self.built = False

    @property
    def algorithm(self):
        """str: Indicates the algorithm name.

        """

        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm):
        self._algorithm = algorithm

    @property
    def hyperparams(self):
        """dict: Contains the key-value parameters
            to meta-heuristics.

        """

        return self._hyperparams

    @hyperparams.setter
    def hyperparams(self, hyperparams):
        if not isinstance(hyperparams, dict):
            raise e.TypeError('`hyperparams` should be a dictionary')

        self._hyperparams = hyperparams

    @property
    def built(self):
        """bool: Indicates whether the optimizer is built.

        """

        return self._built

    @built.setter
    def built(self, built):
        self._built = built

    def _update(self):
        """Updates the agents' position array.

        As each optimizer child can have a different procedure of update,
        you will need to implement it directly on child's class.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError

    @d.pre_evaluation
    def _evaluate(self, space, function):
        """Evaluates the search space according to the objective function.

        If you need a specific evaluate method, please re-implement it on child's class.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object serving as an objective function.

        """

        # Iterates through all agents
        for agent in space.agents:
            # Calculate sthe fitness value of current agent
            agent.fit = function(agent.position)

            # If agent's fitness is better than global fitness
            if agent.fit < space.best_agent.fit:
                # Makes a deep copy of agent's position to the best agent
                space.best_agent.position = copy.deepcopy(agent.position)

                # Also, copies its fitness from agent's fitness
                space.best_agent.fit = copy.deepcopy(agent.fit)

    def run(self, space, function, store_best_only=False, pre_evaluation=None):
        """Runs the optimization pipeline.

        As each optimizer child can have a different optimization pipeline,
        you will need to implement it directly on child's class.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (bool): If True, only the best agent of each iteration is stored in History.
            pre_evaluation (callable): Method to be executed before evaluating the `function` being optimized.

        Raises:
            NotImplementedError.

        """

        raise NotImplementedError

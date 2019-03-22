import copy

import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Optimizer:
    """An Optimizer class that will serve as meta-heuristic
        techniques' parent.

    """

    def __init__(self, algorithm=''):
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
        """str: A string indicating the algorithm name.
        
        """

        return self._algorithm

    @property
    def hyperparams(self):
        """dict: A dictionary containing key-value parameters
            to meta-heuristics.

        """

        return self._hyperparams

    @hyperparams.setter
    def hyperparams(self, hyperparams):
        self._hyperparams = hyperparams

    @property
    def built(self):
        """bool: A boolean to indicate whether the optimizer is built.

        """

        return self._built

    @built.setter
    def built(self, built):
        self._built = built

    def _update(self):
        """Updates the agents' position array.

        As each optimizer child can have a different
        procedure of update, you will need to implement
        it directly on child's class.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError

    def _evaluate(self, space, function):
        """Evaluates the search space according to the objective function.

        If you need a specific evaluate method, please re-implement it on child's class.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that
                will be used as the objective function.

        """

        # Iterate through all agents
        for agent in space.agents:
            # Calculate the fitness value of current agent
            agent.fit = function.pointer(agent.position)

            # If agent's fitness is better than global fitness
            if agent.fit < space.best_agent.fit:
                # Makes a deep copy of current agent to the best agent
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)

    def run(self):
        """Runs the optimization pipeline.
        
        As each optimizer child can have a different
        optimization pipeline, you will need to implement
        it directly on child's class.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError

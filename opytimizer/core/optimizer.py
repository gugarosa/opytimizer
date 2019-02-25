import copy

import opytimizer.utils.common as c
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

    @hyperparams.setter
    def hyperparams(self, hyperparams):
        self._hyperparams = hyperparams

    @property
    def built(self):
        """A boolean to indicate whether the optimizer is built.
        """

        return self._built

    @built.setter
    def built(self, built):
        self._built = built

    def _update(self, agents, best_agent):
        """Updates the agents' position array.
        As each optimizer child can have a different
        procedure of update, you will need to implement
        it directly on child's class.

        Args:
            agents (list): A list of agents that will be updated.
            best_agent (Agent): Global best agent.

        """

        raise NotImplementedError

    def _evaluate(self, space, function):
        """Evaluates the search space according to the objective function.
        If you need a specific evaluate method, please re-implement it on child's class.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        """

        # Iterate through all agents
        for agent in space.agents:
            # Calculate the fitness value of current agent
            fit = function.pointer(agent.position)

            # If fitness is better than agent's best fit
            if fit < agent.fit:
                # Updates its current fitness to the newer one
                agent.fit = fit

            # If agent's fitness is better than global fitness
            if agent.fit < space.best_agent.fit:
                # Makes a depp copy of current agent to the best agent
                space.best_agent = copy.deepcopy(agent)

    def run(self, space, function):
        """Runs the optimization pipeline.
        If you need a specific run method, please re-implement it on child's class.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        """

        # Initial search space evaluation
        self._evaluate(space, function)

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents, space.best_agent)

            # Checking if agents meets the bounds limits
            c.check_bound_limits(space.agents, space.lb, space.ub)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

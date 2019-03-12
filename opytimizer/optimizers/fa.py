import copy

import numpy as np
import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class FA(Optimizer):
    """A FA class, inherited from Optimizer.

    This will be the designed class to define FA-related
    variables and methods.

    References:
        X.-S. Yang. Firefly algorithms for multimodal optimization. International symposium on stochastic algorithms (2009).

    Attributes:
        alpha (float): Randomization parameter.
        beta (float): Attractiveness.
        gamma (float): Light absorption coefficient.

    Methods:
        _build(hyperparams): Sets an external function point to a class attribute.
        _update(self, agents, best_agent, function): Updates the agents according to firefly algorithm.
        run(space, function): Runs the optimization pipeline.

    """

    def __init__(self, hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): An hyperparams dictionary containing key-value
                parameters to meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> FA.')

        # Override its parent class with the receiving hyperparams
        super(FA, self).__init__(algorithm='FA')

        # Randomization parameter
        self._alpha = 0.2

        # Attractiveness
        self._beta = 1.0

        # Light absorption coefficient
        self._gamma = 1.0

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def alpha(self):
        """Randomization parameter.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def beta(self):
        """Attractiveness parameter.

        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta

    @property
    def gamma(self):
        """Light absorption coefficient.

        """

        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    def _build(self, hyperparams):
        """This method will serve as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            hyperparams (dict): An hyperparams dictionary containing key-value
                parameters to meta-heuristics.

        """

        logger.debug('Running private method: build().')

        # We need to save the hyperparams object for faster looking up
        self.hyperparams = hyperparams

        # If one can find any hyperparam inside its object,
        # set them as the ones that will be used
        if hyperparams:
            if 'alpha' in hyperparams:
                self.alpha = hyperparams['alpha']
            if 'beta' in hyperparams:
                self.beta = hyperparams['beta']
            if 'gamma' in hyperparams:
                self.gamma = hyperparams['gamma']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: alpha = {self.alpha}, beta = {self.beta}, gamma = {self.gamma} | Built: {self.built}.')

    def _update(self, agents, best_agent, function):
        """Method that wraps Firefly Algorithm over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.

        """

        return True

    def run(self, space, function):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Initial search space evaluation
        self._evaluate(space, function)

        # We will define a History object for further dumping
        history = h.History()

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents, space.best_agent, function)

            # Checking if agents meets the bounds limits
            space.check_bound_limits(space.agents, space.lb, space.ub)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            # Every iteration, we need to dump the current space agents
            history.dump(space.agents)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

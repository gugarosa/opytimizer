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

    """

    def __init__(self, algorithm='FA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): A string holding optimizer's algorithm name.
            hyperparams (dict): An hyperparams dictionary containing key-value
                parameters to meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> FA.')

        # Override its parent class with the receiving hyperparams
        super(FA, self).__init__(algorithm=algorithm)

        # Randomization parameter
        self._alpha = 0.5

        # Attractiveness
        self._beta = 0.2

        # Light absorption coefficient
        self._gamma = 1.0

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def alpha(self):
        """float: Randomization parameter.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def beta(self):
        """float: Attractiveness parameter.

        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta

    @property
    def gamma(self):
        """float: Light absorption coefficient.

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

    def _update(self, agents, best_agent, function, n_iterations):
        """Method that wraps Firefly Algorithm over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            n_iterations (int): Maximum number of iterations.

        """

        # Calculating current iteration delta
        delta = 1 - ((10 ** -4) / 0.9) ** (1 / n_iterations)

        # Applying update to alpha parameter
        self.alpha *= (1 - delta)

        # We copy a temporary list for iterating purposes
        temp_agents = copy.deepcopy(agents)

        # Iterating through 'i' agents
        for agent in agents:
            # Iterating through 'j' agents
            for temp in temp_agents:
                # Distance is calculated by an euclidean distance between 'i' and 'j' (Equation 8)
                distance = (np.linalg.norm(
                    agent.position - temp.position)) ** 2
                # If 'i' fit is bigger than 'j' fit
                if (agent.fit > temp.fit):
                    # Recalculate the attractiveness (Equation 6)
                    beta = self.beta * np.exp(-self.gamma * distance)

                    # Generates a random uniform distribution
                    r1 = r.generate_uniform_random_number()

                    # Updates agent's position (Equation 9)
                    agent.position = beta * \
                        (temp.position + agent.position) + \
                        self.alpha * (r1 - 0.5)

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
            self._update(space.agents, space.best_agent,
                         function, space.n_iterations)

            # Checking if agents meets the bounds limits
            space.check_bound_limits(space.agents, space.lb, space.ub)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            # Every iteration, we need to dump the current space agents
            history.dump(space.agents, space.best_agent)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

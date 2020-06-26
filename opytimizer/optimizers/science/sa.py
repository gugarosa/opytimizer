import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class SA(Optimizer):
    """A SA class, inherited from Optimizer.

    This is the designed class to define SA-related
    variables and methods.

    References:
        A. Khachaturyan, S. Semenovsovskaya and B. Vainshtein.
        The thermodynamic approach to the structure analysis of crystals.
        Acta Crystallographica (1981).

    """

    def __init__(self, algorithm='SA', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> SA.')

        # Override its parent class with the receiving hyperparams
        super(SA, self).__init__(algorithm)

        # System's temperature
        self.T = 100

        # Temperature decay
        self.beta = 0.999

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def T(self):
        """float: System's temperature.

        """

        return self._T

    @T.setter
    def T(self, T):
        if not (isinstance(T, float) or isinstance(T, int)):
            raise e.TypeError('`T` should be a float or integer')
        if T < 0:
            raise e.ValueError('`T` should be >= 0')

        self._T = T

    @property
    def beta(self):
        """float: Temperature decay.

        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        if not (isinstance(beta, float) or isinstance(beta, int)):
            raise e.TypeError('`beta` should be a float or integer')
        if beta < 0:
            raise e.ValueError('`beta` should be >= 0')

        self._beta = beta

    def _build(self, hyperparams):
        """This method serves as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.debug('Running private method: build().')

        # We need to save the hyperparams object for faster looking up
        self.hyperparams = hyperparams

        # If one can find any hyperparam inside its object,
        # set them as the ones that will be used
        if hyperparams:
            if 'T' in hyperparams:
                self.T = hyperparams['T']
            if 'beta' in hyperparams:
                self.beta = hyperparams['beta']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | '
            f'Hyperparameters: T = {self.T}, beta = {self.beta} | Built: {self.built}.')

    def _update(self, agents, function):
        """Method that wraps Simulated Annealing over all agents and variables.

        Args:
            agents (list): List of agents.
            function (Function): A function object.

        """

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # Mimics its position
            a = copy.deepcopy(agent)

            # Generating a random noise from a gaussian distribution
            noise = r.generate_gaussian_random_number(
                0, 0.1, size=((agent.n_variables, agent.n_dimensions)))

            # Applying the noise
            a.position += noise

            # Check agent limits
            a.clip_limits()

            # Calculates the fitness for the temporary position
            a.fit = function(a.position)

            # Generates an uniform random number
            r1 = r.generate_uniform_random_number()

            # If new fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Copy its position to the agent
                agent.position = copy.deepcopy(a.position)

                # And also copy its fitness
                agent.fit = copy.deepcopy(a.fit)

            # Checks if state should be updated or not
            elif r1 < np.exp(-(a.fit - agent.fit) / self.T):
                # Copy its position to the agent
                agent.position = copy.deepcopy(a.position)

                # And also copy its fitness
                agent.fit = copy.deepcopy(a.fit)

        # Decay the temperature
        self.T *= self.beta

    def run(self, space, function, store_best_only=False, pre_evaluation=None):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (bool): If True, only the best agent of each iteration is stored in History.
            pre_evaluation (callable): This function is executed before evaluating the function being optimized.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Initial search space evaluation
        self._evaluate(space, function, hook=pre_evaluation)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.file(f'Iteration {t+1}/{space.n_iterations}')

                # Updating agents
                self._update(space.agents, function)

                # Checking if agents meets the bounds limits
                space.clip_limits()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, hook=pre_evaluation)

                # Every iteration, we need to dump agents and best agent
                history.dump(agents=space.agents, best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.file(f'Fitness: {space.best_agent.fit}')
                logger.file(f'Position: {space.best_agent.position}')

        return history

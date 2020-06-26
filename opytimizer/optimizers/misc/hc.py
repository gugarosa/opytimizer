from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class HC(Optimizer):
    """A HC class, inherited from Optimizer.

    This is the designed class to define HC-related
    variables and methods.

    References:
        S. Skiena. The Algorithm Design Manual (2010).

    """

    def __init__(self, algorithm='HC', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> HC.')

        # Override its parent class with the receiving hyperparams
        super(HC, self).__init__(algorithm)

        # Mean of noise distribution
        self.r_mean = 0

        # Variance of noise distribution
        self.r_var = 0.1

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def r_mean(self):
        """float: Mean of noise distribution.

        """

        return self._r_mean

    @r_mean.setter
    def r_mean(self, r_mean):
        if not (isinstance(r_mean, float) or isinstance(r_mean, int)):
            raise e.TypeError('`r_mean` should be a float or integer')

        self._r_mean = r_mean

    @property
    def r_var(self):
        """float: Variance of noise distribution.

        """

        return self._r_var

    @r_var.setter
    def r_var(self, r_var):
        if not (isinstance(r_var, float) or isinstance(r_var, int)):
            raise e.TypeError('`r_var` should be a float or integer')
        if r_var < 0:
            raise e.ValueError('`r_var` should be >= 0')

        self._r_var = r_var

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
            if 'r_mean' in hyperparams:
                self.r_mean = hyperparams['r_mean']
            if 'r_var' in hyperparams:
                self.r_var = hyperparams['r_var']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | '
            f'Hyperparameters: r_mean = {self.r_mean}, r_var = {self.r_var} | '
            f'Built: {self.built}.')

    def _update(self, agents):
        """Method that wraps Hill Climbing over all agents and variables.

        Args:
            agents (list): List of agents.

        """

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # Creates a gaussian noise vector
            noise = r.generate_gaussian_random_number(
                self.r_mean, self.r_var, size=(agent.n_variables, agent.n_dimensions))

            # Updating agent's position
            agent.position += noise

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
                self._update(space.agents)

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

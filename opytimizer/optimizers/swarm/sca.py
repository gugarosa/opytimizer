import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class SCA(Optimizer):
    """A SCA class, inherited from Optimizer.

    This is the designed class to define SCA-related
    variables and methods.

    References:
        S. Mirjalili. SCA: A Sine Cosine Algorithm for solving optimization problems.
        Knowledge-Based Systems (2016).

    """

    def __init__(self, algorithm='SCA', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> SCA.')

        # Override its parent class with the receiving hyperparams
        super(SCA, self).__init__(algorithm)

        # Minimum function range
        self.r_min = 0

        # Maximum function range
        self.r_max = 2

        # Constant for defining the next position's region
        self.a = 3

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def r_min(self):
        """float: Minimum function range.

        """

        return self._r_min

    @r_min.setter
    def r_min(self, r_min):
        if not (isinstance(r_min, float) or isinstance(r_min, int)):
            raise e.TypeError('`r_min` should be a float or integer')
        if r_min < 0:
            raise e.ValueError('`r_min` should be >= 0')

        self._r_min = r_min

    @property
    def r_max(self):
        """float: Maximum function range.

        """

        return self._r_max

    @r_max.setter
    def r_max(self, r_max):
        if not (isinstance(r_max, float) or isinstance(r_max, int)):
            raise e.TypeError('`r_max` should be a float or integer')
        if r_max < 0:
            raise e.ValueError('`r_max` should be >= 0')
        if r_max < self.r_min:
            raise e.ValueError('`r_max` should be >= `r_min`')

        self._r_max = r_max

    @property
    def a(self):
        """float: Loudness parameter.

        """

        return self._a

    @a.setter
    def a(self, a):
        if not (isinstance(a, float) or isinstance(a, int)):
            raise e.TypeError('`a` should be a float or integer')
        if a < 0:
            raise e.ValueError('`a` should be >= 0')

        self._a = a

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
            if 'r_min' in hyperparams:
                self.r_min = hyperparams['r_min']
            if 'r_max' in hyperparams:
                self.r_max = hyperparams['r_max']
            if 'a' in hyperparams:
                self.a = hyperparams['a']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | '
            f'Hyperparameters: r_min = {self.r_min}, r_max = {self.r_max}, a = {self.a} | '
            f'Built: {self.built}.')

    def _update_position(self, agent_position, best_position, r1, r2, r3, r4):
        """Updates a single particle position (over a single variable).

        Args:
            agent_position (np.array): Agent's current position.
            best_position (np.array): Global best position.
            r1 (float): Controls the next position's region.
            r2 (float): Defines how far the movement should be.
            r3 (float): Random weight for emphasizing or deemphasizing the movement.
            r4 (float): Random number to decide whether sine or cosine should be used.

        Returns:
            A new position based on SCA's paper equation 3.3.

        """

        # If random number is smaller than threshold
        if r4 < 0.5:
            # Updates the position using sine
            new_position = agent_position + r1 * \
                np.sin(r2) * np.fabs(r3 * best_position - agent_position)

        # If the random number is bigger than threshold
        else:
            # Updates the posistion using cosine
            new_position = agent_position + r1 * \
                np.cos(r2) * np.fabs(r3 * best_position - agent_position)

        return new_position

    def _update(self, agents, best_agent, function, iteration, n_iterations):
        """Method that wraps Bat Algorithm over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            iteration (int): Current iteration value.
            n_iterations (int): Maximum number of iterations.

        """

        # Adaptively changing the r1 parameter, which controls the next position's region
        r1 = self.a - (iteration * self.a / n_iterations)

        # The r2 parameter defines how far the movement should be
        r2 = r.generate_uniform_random_number(0, 2 * np.pi)

        # A random weight for emphasizing or deemphasizing the movement
        r3 = r.generate_uniform_random_number(self.r_min, self.r_max)

        # A random number to decide whether sine or cosine should be used
        r4 = r.generate_uniform_random_number()

        # Iterate through all agents
        for agent in agents:
            # Updates agent's position
            agent.position = self._update_position(
                agent.position, best_agent.position, r1, r2, r3, r4)

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
                self._update(space.agents, space.best_agent, function, t, space.n_iterations)

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

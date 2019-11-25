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
        super(HC, self).__init__(algorithm=algorithm)

        # Type of noise
        self.type = 'gaussian'

        # Minimum noise range
        self.r_min = 0

        # Maximum noise range
        self.r_max = 0.1

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def type(self):
        """str: Type of noise.

        """

        return self._type

    @type.setter
    def type(self, type):
        if not ((type == 'gaussian') or (type == 'uniform')):
            raise e.TypeError('`type` should be `gaussian` or `uniform`')

        self._type = type

    @property
    def r_min(self):
        """float: Minimum noise range.

        """

        return self._r_min

    @r_min.setter
    def r_min(self, r_min):
        if not (isinstance(r_min, float) or isinstance(r_min, int)):
            raise e.TypeError('`r_min` should be a float or integer')

        self._r_min = r_min

    @property
    def r_max(self):
        """float: Maximum noise range.

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
            if 'type' in hyperparams:
                self.type = hyperparams['type']
            if 'r_min' in hyperparams:
                self.r_min = hyperparams['r_min']
            if 'r_max' in hyperparams:
                self.r_max = hyperparams['r_max']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: type = {self.type}, r_min = {self.r_min}, r_max = {self.r_max} | Built: {self.built}.')

    def _update(self, agents):
        """Method that wraps Hill Climbing over all agents and variables.

        Args:
            agents (list): List of agents.

        """

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # If type of noise is gaussian
            if self.type == 'gaussian':
                # Creates a gaussian noise vector
                noise = r.generate_gaussian_random_number(
                    self.r_min, self.r_max, size=(agent.n_variables, agent.n_dimensions))

            # If type of noise is uniform
            if self.type == 'uniform':
                # Creates an uniform noise vector
                noise = r.generate_uniform_random_number(
                    self.r_min, self.r_max, size=(agent.n_variables, agent.n_dimensions))

            # Updating agent's position
            agent.position += noise

    def run(self, space, function, store_best_only=False, pre_evaluation_hook=None):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (boolean): If True, only the best agent of each iteration is stored in History.
            pre_evaluation_hook (function): A function that receives the optimizer, space and function
                and returns None. This function is executed before evaluating the function being optimized.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Check if there is a pre-evaluation hook
        if pre_evaluation_hook:
            # Applies the hook
            pre_evaluation_hook(self, space, function)

        # Initial search space evaluation
        self._evaluate(space, function)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents)

            # Checking if agents meets the bounds limits
            space.check_limits()

            # Check if there is a pre-evaluation hook
            if pre_evaluation_hook:
                # Applies the hook
                pre_evaluation_hook(self, space, function)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            # Every iteration, we need to dump agents and best agent
            history.dump(agents=space.agents, best_agent=space.best_agent)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

import copy

from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class IWO(Optimizer):
    """An IWO class, inherited from Optimizer.

    This is the designed class to define IWO-related
    variables and methods.

    References:
        A. R. Mehrabian and C. Lucas. A novel numerical optimization algorithm inspired from weed colonization.
        Ecological informatics (2006).
        
    """

    def __init__(self, algorithm='IWO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Override its parent class with the receiving hyperparams
        super(IWO, self).__init__(algorithm)

        # Minimum number of seeds
        self.min_seeds = 0

        # Maximum number of seeds
        self.max_seeds = 5

        # Exponent to calculate the Spatial Dispersal
        self.e = 2

        # Final standard deviation
        self.final_sigma = 0.001

        # Initial standard deviation
        self.init_sigma = 3

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def min_seeds(self):
        """int: Minimum number of seeds.

        """

        return self._min_seeds

    @min_seeds.setter
    def min_seeds(self, min_seeds):
        if not isinstance(min_seeds, int):
            raise e.TypeError('`min_seeds` should be an integer')
        if min_seeds < 0:
            raise e.ValueError('`min_seeds` should be >= 0')

        self._min_seeds = min_seeds

    @property
    def max_seeds(self):
        """int: Maximum number of seeds.

        """

        return self._max_seeds

    @max_seeds.setter
    def max_seeds(self, max_seeds):
        if not isinstance(max_seeds, int):
            raise e.TypeError('`max_seeds` should be an integer')
        if max_seeds < self.min_seeds:
            raise e.ValueError('`max_seeds` should be >= `min_seeds`')

        self._max_seeds = max_seeds

    @property
    def e(self):
        """float: Exponent used to calculate the Spatial Dispersal.

        """

        return self._e

    @e.setter
    def e(self, e):
        if not (isinstance(e, float) or isinstance(e, int)):
            raise e.TypeError('`e` should be a float or integer')
        if e < 0:
            raise e.ValueError('`e` should be >= 0')

        self._e = e

    @property
    def final_sigma(self):
        """float: Final standard deviation.

        """

        return self._final_sigma

    @final_sigma.setter
    def final_sigma(self, final_sigma):
        if not (isinstance(final_sigma, float) or isinstance(final_sigma, int)):
            raise e.TypeError('`final_sigma` should be a float or integer')
        if final_sigma < 0:
            raise e.ValueError('`final_sigma` should be >= 0')


        self._final_sigma = final_sigma

    @property
    def init_sigma(self):
        """float: Initial standard deviation.

        """

        return self._init_sigma

    @init_sigma.setter
    def init_sigma(self, init_sigma):
        if not (isinstance(init_sigma, float) or isinstance(init_sigma, int)):
            raise e.TypeError('`init_sigma` should be a float or integer')
        if init_sigma < 0:
            raise e.ValueError('`init_sigma` should be >= 0')
        if init_sigma < self.final_sigma:
            raise e.ValueError('`init_sigma` should be >= `final_sigma`')

        self._init_sigma = init_sigma

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
            if 'min_seeds' in hyperparams:
                self.min_seeds = hyperparams['min_seeds']
            if 'max_seeds' in hyperparams:
                self.max_seeds = hyperparams['max_seeds']
            if 'e' in hyperparams:
                self.e = hyperparams['e']
            if 'final_sigma' in hyperparams:
                self.final_sigma = hyperparams['final_sigma']
            if 'init_sigma' in hyperparams:
                self.init_sigma = hyperparams['init_sigma']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | '
            f'Hyperparameters: min_seeds = {self.min_seeds}, max_seeds = {self.max_seeds}, e = {self.e}, '
            f'init_sigma = {self.init_sigma}, final_sigma = {self.final_sigma} | '
            f'Built: {self.built}.')

    def _spatial_dispersal(self, iteration, n_iterations):
        """Calculates the Spatial Dispersal coefficient.

        Args:
            iteration (int): Current iteration number.
            n_iterations (int): Maximum number of iterations.

        """

        # Calculating the iteration coefficient
        coef = ((n_iterations - iteration) ** self.e) / ((n_iterations + c.EPSILON) ** self.e)

        # Updating the Spatial Dispersial
        self.sigma = coef * (self.init_sigma - self.final_sigma) + self.final_sigma

    def _produce_offspring(self, agent, function):
        """Reproduces and flowers a seed into a new offpsring.

        Args:
            agent (Agent): An agent instance to be reproduced.
            function (Function): A Function object that will be used as the objective function.

        Returns:
            An evolved offspring.

        """

        # Makea a deepcopy on selected agent
        a = copy.deepcopy(agent)

        # For every possible decision variable
        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            # Updates its position
            a.position[j] += self.sigma * r.generate_uniform_random_number(lb, ub, a.n_dimensions)

        # Clips its limits
        a.clip_limits()

        # Calculates its fitness
        a.fit = function(a.position)

        return a

    def _update(self, agents, n_agents, function):
        """Method that wraps offsprings generations over all agents and variables.

        Args:
            agents (list): List of agents.
            n_agents (int): Number of possible agents in the space.
            function (Function): A Function object that will be used as the objective function.

        Returns:
            A new population with more fitted individuals.

        """

        # Creating a list for the produced offsprings
        offsprings = []

        # Sorting agents
        agents.sort(key=lambda x: x.fit)

        # Iterate through all agents
        for agent in agents:
            # Calculate the seeding ratio based on its fitness
            ratio = (agent.fit - agents[-1].fit) / (agents[0].fit - agents[-1].fit + c.EPSILON)

            # Calculates the number of produced seeds
            n_seeds = int(self.min_seeds + (self.max_seeds - self.min_seeds) * ratio)

            # For every seed
            for _ in range(n_seeds):
                # Reproduces and flowers the seed into a new agent
                a = self._produce_offspring(agent, function)

                # Appends the agent to the offsprings
                offsprings.append(a)

        # Joins both populations
        agents += offsprings

        # Performs a new sort on the merged population
        agents.sort(key=lambda x: x.fit)

        return agents[:n_agents]

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

                # Calculates the current Spatial Dispersal
                self._spatial_dispersal(t, space.n_iterations)

                # Updating agents
                space.agents = self._update(space.agents, space.n_agents, function)

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

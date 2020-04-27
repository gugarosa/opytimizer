import copy

import opytimizer.math.distribution as d
import opytimizer.math.random as r
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
        

    """

    def __init__(self, algorithm='IWO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Override its parent class with the receiving hyperparams
        super(IWO, self).__init__(algorithm)

        self.min_seeds = 0

        self.max_seeds = 5

        self.init_sigma = 0.5

        self.end_sigma = 0.001

        # Lévy flight control parameter
        self.beta = 1.5

        # Lévy flight scaling factor
        self.eta = 0.2

        # Probability of local pollination
        self.p = 0.8

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def beta(self):
        """float: Lévy flight control parameter.

        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        if not (isinstance(beta, float) or isinstance(beta, int)):
            raise e.TypeError('`beta` should be a float or integer')
        if beta < 0:
            raise e.ValueError('`beta` should be >= 0')

        self._beta = beta

    @property
    def eta(self):
        """float: Lévy flight scaling factor.

        """

        return self._eta

    @eta.setter
    def eta(self, eta):
        if not (isinstance(eta, float) or isinstance(eta, int)):
            raise e.TypeError('`eta` should be a float or integer')
        if eta < 0:
            raise e.ValueError('`eta` should be >= 0')

        self._eta = eta

    @property
    def p(self):
        """float: Probability of local pollination.

        """

        return self._p

    @p.setter
    def p(self, p):
        if not (isinstance(p, float) or isinstance(p, int)):
            raise e.TypeError('`p` should be a float or integer')
        if p < 0 or p > 1:
            raise e.ValueError('`p` should be between 0 and 1')

        self._p = p

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
            if 'beta' in hyperparams:
                self.beta = hyperparams['beta']
            if 'eta' in hyperparams:
                self.eta = hyperparams['eta']
            if 'p' in hyperparams:
                self.p = hyperparams['p']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | '
            f'Hyperparameters: beta = {self.beta}, eta = {self.eta}, p = {self.p} | '
            f'Built: {self.built}.')

    def _update(self, agents, n_agents, function):
        """Method that wraps global and local pollination updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.

        """

        new_agents = []

        # Sorting agents
        agents.sort(key=lambda x: x.fit)

        #
        for agent in agents:
            #
            ratio = (agent.fit - agents[-1].fit) / (agents[0].fit - agents[-1].fit)

            # print(ratio)

            #
            n_seeds = int(self.min_seeds + (self.max_seeds - self.min_seeds) * ratio)

            #
            for _ in range(n_seeds):
                #
                a = copy.deepcopy(agent)

                for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
                    a.position[j] += self.sigma * r.generate_uniform_random_number(lb, ub, a.n_dimensions)

                #
                a.fit = function.pointer(a.position)

                #
                new_agents.append(a)

        #
        agents += new_agents

        #
        agents.sort(key=lambda x: x.fit)

        #
        return agents[:n_agents]


    def run(self, space, function, store_best_only=False, pre_evaluation_hook=None):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (bool): If True, only the best agent of each iteration is stored in History.
            pre_evaluation_hook (callable): This function is executed before evaluating the function being optimized.

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

            #
            self.sigma = ((space.n_iterations - t) / (space.n_iterations - 1)) ** 2 * (self.init_sigma - self.end_sigma) + self.end_sigma

            # Updating agents
            space.agents = self._update(space.agents, space.n_agents, function)

            # Checking if agents meets the bounds limits
            space.clip_limits()

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

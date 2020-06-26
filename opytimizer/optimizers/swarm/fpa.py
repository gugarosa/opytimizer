import copy

from tqdm import tqdm

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class FPA(Optimizer):
    """A FPA class, inherited from Optimizer.

    This is the designed class to define FPA-related
    variables and methods.

    References:
        X.-S. Yang. Flower pollination algorithm for global optimization.
        International conference on unconventional computing and natural computation (2012).

    """

    def __init__(self, algorithm='FPA', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Override its parent class with the receiving hyperparams
        super(FPA, self).__init__(algorithm)

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

    def _global_pollination(self, agent_position, best_position):
        """Updates the agent's position based on a global pollination (Lévy's flight).

        Args:
            agent_position (np.array): Agent's current position.
            best_position (np.array): Best agent's current position.

        Returns:
            A new position based on FPA's paper equation 1.

        """

        # Generates a Lévy distribution
        step = d.generate_levy_distribution(self.beta)

        # Calculates the global pollination
        global_pollination = self.eta * step * (best_position - agent_position)

        # Calculates the new position based on previous global pollination
        new_position = agent_position + global_pollination

        return new_position

    def _local_pollination(self, agent_position, k_position, l_position, epsilon):
        """Updates the agent's position based on a local pollination.

        Args:
            agent_position (np.array): Agent's current position.
            k_position (np.array): Agent's (index k) current position.
            l_position (np.array): Agent's (index l) current position.
            epsilon (float): An uniform random generated number.

        Returns:
            A new position based on FPA's paper equation 3.

        """

        # Calculates the local pollination
        local_pollination = epsilon * (k_position - l_position)

        # Calculates the new position based on previous local pollination
        new_position = agent_position + local_pollination

        return new_position

    def _update(self, agents, best_agent, function):
        """Method that wraps global and local pollination updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.

        """

        # Iterate through all agents
        for agent in agents:
            # Creates a temporary agent
            a = copy.deepcopy(agent)

            # Generating an uniform random number
            r1 = r.generate_uniform_random_number()

            # Check if generated random number is bigger than probability
            if r1 > self.p:
                # Update a temporary position according to global pollination
                a.position = self._global_pollination(
                    agent.position, best_agent.position)

            else:
                # Generates an uniform random number
                epsilon = r.generate_uniform_random_number()

                # Generates an index for flower k
                k = int(r.generate_uniform_random_number(0, len(agents)-1))

                # Generates an index for flower l
                l = int(r.generate_uniform_random_number(0, len(agents)-1))

                # Update a temporary position according to local pollination
                a.position = self._local_pollination(
                    agent.position, agents[k].position, agents[l].position, epsilon)

            # Check agent limits
            a.clip_limits()

            # Calculates the fitness for the temporary position
            a.fit = function(a.position)

            # If new fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Copy its position to the agent
                agent.position = copy.deepcopy(a.position)

                # And also copy its fitness
                agent.fit = copy.deepcopy(a.fit)

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
                self._update(space.agents, space.best_agent, function)

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

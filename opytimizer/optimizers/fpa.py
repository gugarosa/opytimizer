import copy

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class FPA(Optimizer):
    """A FPA class, inherited from Optimizer.

    This will be the designed class to define FPA-related
    variables and methods.

    References:
        X.-S. Yang. Flower pollination algorithm for global optimization. International conference on unconventional computing and natural computation (2012).

    """

    def __init__(self, algorithm='FPA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): A string holding optimizer's algorithm name.
            hyperparams (dict): An hyperparams dictionary containing key-value
                parameters to meta-heuristics.

        """

        # Override its parent class with the receiving hyperparams
        super(FPA, self).__init__(algorithm=algorithm)

        # Lévy flight control parameter
        self._beta = 1.5

        # Lévy flight scaling factor
        self._eta = 0.2

        # Probability of local pollination
        self._p = 0.8

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
        self._beta = beta

    @property
    def eta(self):
        """float: Lévy flight scaling factor.

        """

        return self._eta

    @eta.setter
    def eta(self, eta):
        self._eta = eta

    @property
    def p(self):
        """float: Probability of local pollination.

        """

        return self._p

    @p.setter
    def p(self, p):
        self._p = p

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
            f'Algorithm: {self.algorithm} | Hyperparameters: beta = {self.beta}, eta = {self.eta}, p = {self.p} | Built: {self.built}.')

    def _global_pollination(self, agent_position, best_position):
        """Updates the agent's position based on a global pollination (Lévy's flight).

        Args:
            agent_position (float): Agent's current position.
            best_position (float): Best agent's current position.

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
            agent_position (float): Agent's current position.
            k_position (float): Agent's (index k) current position.
            l_position (float): Agent's (index l) current position.
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
            # Generating an uniform random number
            r1 = r.generate_uniform_random_number()

            # Check if generated random number is bigger than probability
            if r1 > self.p:
                # Update a temporary position according to global pollination
                temp_position = self._global_pollination(
                    agent.position, best_agent.position)

            else:
                # Generates an uniform random number
                epsilon = r.generate_uniform_random_number()

                # Generates an index for flower k
                k = int(r.generate_uniform_random_number(0, len(agents)-1))

                # Generates an index for flower l
                l = int(r.generate_uniform_random_number(0, len(agents)-1))

                # Update a temporary position according to local pollination
                temp_position = self._local_pollination(
                    agent.position, agents[k].position, agents[l].position, epsilon)

            # Calculates the fitness for the temporary position
            fit = function.pointer(temp_position)

            # If new fitness is better than agent's fitness
            if fit < agent.fit:
                # Copy its position to the agent
                agent.position = copy.deepcopy(temp_position)

                # And also copy its fitness
                agent.fit = copy.deepcopy(fit)

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
            history.dump(space.agents, space.best_agent)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

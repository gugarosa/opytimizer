import copy

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class FPA(Optimizer):
    """A FPA class, inherited from Optimizer.

    This will be the designed class to define FPA-related
    variables and methods.

    References:
        Yang, X.-S. Unconventional Computation and Natural Computation (2012). 

    Properties:
        beta (float): Lévy flight control parameter.
        eta (float): Lévy flight scaling factor.
        p (float): Probability of local pollination.

    Methods:
        _build(hyperparams): Sets an external function point to a class attribute.
        _global_pollination(agent_position, best_position): Updates the agent's position based on a global pollination (Lévy's flight).
        _local_pollination(agent_position, k_position, l_position, epsilon): Updates the agent's position based on a local pollination.
        _update(agents, best_agent): Updates the agents' position array.

    """

    def __init__(self, hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): An hyperparams dictionary containing key-value
            parameters to meta-heuristics.

        """

        # Override its parent class with the receiving hyperparams
        super(FPA, self).__init__(algorithm='FPA')

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
        """Lévy flight control parameter.
        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta

    @property
    def eta(self):
        """Lévy flight scaling factor.
        """

        return self._eta

    @eta.setter
    def eta(self, eta):
        self._eta = eta

    @property
    def p(self):
        """Probability of local pollination.
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

        logger.debug('Running private method: build()')

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
            f'Algorithm: {self.algorithm} | Hyperparameters: beta = {self.beta}, eta = {self.eta}, p = {self.p} | Built: {self.built}')

    def _global_pollination(self, agent_position, best_position):
        """Updates the agent's position based on a global pollination (Lévy's flight).

        Args:
            agent_position (float): Agent's current position.
            best_position (float): Best agent's current position.

        Returns:
            A new position based on FPA's paper global pollination equation.

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
            epsilon (float): An uniform random generated number

        Returns:
            A new position based on FPA's paper local pollination equation.

        """

        # Calculates the local pollination
        local_pollination = epsilon * (k_position - l_position)

        # Calculates the new position based on previous local pollination
        new_position = agent_position + local_pollination

        return new_position

    def _update(self, agents, best_agent):
        """Method that wraps global and local pollination updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.

        """

        # Iterate through all agents
        for agent in agents:
            # Generating an uniform random number
            r1 = r.generate_uniform_random_number(0, 1)

            # Check if generated random number is bigger than probability
            if r1 > self.p:
                # Update each decision variable according to global pollination
                agent.position = self._global_pollination(
                    agent.position, best_agent.position)
            else:
                # Generates an uniform random number
                epsilon = r.generate_uniform_random_number(0, 1)

                # Generates an index for flower k
                k = int(r.generate_uniform_random_number(0, len(agents)-1))

                # Generates an index for flower l
                l = int(r.generate_uniform_random_number(0, len(agents)-1))

                # Update each decision variable according to local pollination
                agent.position = self._local_pollination(
                    agent.position, agents[k].position, agents[l].position, epsilon)

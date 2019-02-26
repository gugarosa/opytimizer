import copy
import operator
import random

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.common as c
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
        _levy_flight(): Updates the agent's position based on a Lévy's flight.
        _local_pollination(): Updates the agent's position based on a local pollination.

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

    def _levy_flight(self, agent_position, best_position):
        """
        """

        step = d.generate_levy_distribution(self.beta, len(agent_position))
        eta_step = list(map(lambda x: x*self.eta, step))
        aux = list(map(operator.mul, eta_step, map(
            operator.sub, best_position, agent_position)))
        levy_flight = list(map(operator.add, agent_position, aux))

        return levy_flight

    def _local_pollination(self, agents, agent_position):
        """
        """

        epsilon = r.generate_uniform_random_number(0, 1)
        flowers = random.sample(agents, 2)
        sub = list(map(operator.sub, flowers[0].position, flowers[1].position))
        sub_epsilon = list(map(lambda x: x*epsilon, sub))
        local_pollination = list(
            map(operator.add, agent_position, sub_epsilon))

        return local_pollination

    def _update(self, agents, best_agent):
        """
        """
        
        # Iterate through all agents
        for i, agent in enumerate(agents):
            if r.generate_uniform_random_number(0, 1) > self.p:
                agent.position = list(map(operator.add, agent.position, self._levy_flight(
                    agent.position, best_agent.position)))
            else:
                agent.position = list(map(
                    operator.add, agent.position, self._local_pollination(agents, agent.position)))

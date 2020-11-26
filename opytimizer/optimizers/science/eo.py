"""Equilibrium Optimizer.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as rnd
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class EO(Optimizer):
    """An EO class, inherited from Optimizer.

    This is the designed class to define EO-related
    variables and methods.

    References:
        A. Faramarzi et al. Equilibrium optimizer: A novel optimization algorithm.
        Knowledge-Based Systems (2020).

    """

    def __init__(self, algorithm='EO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> EO.')

        # Override its parent class with the receiving hyperparams
        super(EO, self).__init__(algorithm)

        # Exploration constant
        self.a1 = 2.0

        # Exploitation constant
        self.a2 = 1.0

        # Generation probability
        self.GP = 0.5

        # Velocity
        self.V = 1.0

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def a1(self):
        """float: Exploration constant.

        """

        return self._a1

    @a1.setter
    def a1(self, a1):
        if not isinstance(a1, (float, int)):
            raise e.TypeError('`a1` should be a float or integer')
        if a1 < 0:
            raise e.ValueError('`a1` should be >= 0')

        self._a1 = a1

    @property
    def a2(self):
        """float: Exploitation constant.

        """

        return self._a2

    @a2.setter
    def a2(self, a2):
        if not isinstance(a2, (float, int)):
            raise e.TypeError('`a2` should be a float or integer')
        if a2 < 0:
            raise e.ValueError('`a2` should be >= 0')

        self._a2 = a2

    @property
    def GP(self):
        """float: Generation probability.

        """

        return self._GP

    @GP.setter
    def GP(self, GP):
        if not isinstance(GP, (float, int)):
            raise e.TypeError('`GP` should be a float or integer')
        if GP < 0 or GP > 1:
            raise e.ValueError('`GP` should be between 0 and 1')

        self._GP = GP

    @property
    def V(self):
        """float: Velocity.

        """

        return self._V

    @V.setter
    def V(self, V):
        if not isinstance(V, (float, int)):
            raise e.TypeError('`V` should be a float or integer')
        if V < 0:
            raise e.ValueError('`V` should be >= 0')

        self._V = V

    def _calculate_equilibrium(self, agents, C):
        """Calculates the equilibrium concentrations.

        Args:
            agents (list): List of agents.
            C (list): List of concentrations to be updated.

        Returns:
            List of equilibrium concentrations.

        """

        # Iterates through all agents
        for agent in agents:
            # If current agent's fitness is smaller than C0
            if agent.fit < C[0].fit:
                # Replaces C0 object
                C[0] = copy.deepcopy(agent)

            # If current agent's fitness is between C0 and C1
            elif agent.fit < C[1].fit:
                # Replaces C1 object
                C[1] = copy.deepcopy(agent)

            # If current agent's fitness is between C1 and C2
            elif agent.fit < C[2].fit:
                # Replaces C2 object
                C[2] = copy.deepcopy(agent)

            # If current agent's fitness is between C2 and C3
            elif agent.fit < C[3].fit:
                # Replaces C3 object
                C[3] = copy.deepcopy(agent)

        return C

    def _average_concentration(self, function, C):
        """Averages the concentrations.

        Args:
            function (Function): A Function object that will be used as the objective function.
            C (list): List of concentrations.

        Returns:
            Averaged concentration.

        """

        # Makes a deepcopy to withhold the future update
        C_avg = copy.deepcopy(C[0])

        # Update the position with concentrations' averager
        C_avg.position = np.mean([c.position for c in C], axis=0)

        # Clips its limits
        C_avg.clip_limits()

        # Re-calculate its fitness
        C_avg.fit = function(C_avg.position)

        return C_avg

    def _update(self, agents, function, C, iteration, n_iterations):
        """Method that wraps Equilibrium Optimizer over all agents and variables.

        Args:
            agents (list): List of agents.
            function (Function): A Function object that will be used as the objective function.
            C (list): List of concentrations.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Calculates the equilibrium concentrations
        C = self._calculate_equilibrium(agents, C)

        # Calculates the average concentration
        C_avg = self._average_concentration(function, C)

        # Makes a pool of both concentrations and their average (eq. 7)
        C_pool = C + [C_avg]

        # Calculates the time (eq. 9)
        t = (1 - iteration / n_iterations) ** (self.a2 * iteration / n_iterations)

        # Iterates through all agents
        for agent in agents:
            # Generates a integer between [0, 5) to select the concentration
            i = rnd.generate_integer_random_number(0, 5)

            # Generates two uniform random vectors (eq. 11)
            r = rnd.generate_uniform_random_number(size=(agent.n_variables, agent.n_dimensions))
            lambd = rnd.generate_uniform_random_number(size=(agent.n_variables, agent.n_dimensions))

            # Calculates the exponential term (eq. 11)
            F = self.a1 * np.sign(r - 0.5) * (np.exp(-lambd * t) - 1)

            # Generates two uniform random numbers
            r1 = rnd.generate_uniform_random_number()
            r2 = rnd.generate_uniform_random_number()

            # If `r2` is bigger than generation probability (eq. 15)
            if r2 >= self.GP:
                # Defines generation control parameter as 0.5 * r1
                GCP = 0.5 * r1

            # If `r2` is smaller than generation probability
            else:
                # Defines generation control parameter as zero
                GCP = 0

            # Calculates the initial generation value (eq. 14)
            G_0 = GCP * (C_pool[i].position - lambd * agent.position)

            # Calculates the generation value (eq. 13)
            G = G_0 * F

            # Updates agent's position (eq. 16)
            agent.position = C_pool[i].position + (
                agent.position - C_pool[i].position) * F + (G / (lambd * self.V)) * (1 - F)

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

        # Creates a list of concentrations (agents)
        C = [copy.deepcopy(space.agents[0]) for _ in range(4)]

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
                self._update(space.agents, function, C, t, space.n_iterations)

                # Checking if agents meet the bounds limits
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

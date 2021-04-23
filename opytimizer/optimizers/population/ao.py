"""Aquila Optimizer.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class AO(Optimizer):
    """An AO class, inherited from Optimizer.

    This is the designed class to define AO-related
    variables and methods.

    References:
        L. Abualigah et al. Aquila Optimizer: A novel meta-heuristic optimization Algorithm.
        Computers & Industrial Engineering (2021).

    """

    def __init__(self, algorithm='AO', params=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> AO.')

        # Override its parent class with the receiving params
        super(AO, self).__init__(algorithm)

        # First exploitation adjustment coefficient
        self.alpha = 0.1

        # Second exploitation adjustment coefficient
        self.delta = 0.1

        # Number of search cycles
        self.n_cycles = 10

        # Cycle regularizer
        self.U = 0.00565

        # Angle regularizer
        self.w = 0.005

        # Now, we need to build this class up
        self._build(params)

        logger.info('Class overrided.')

    @property
    def alpha(self):
        """float: First exploitation adjustment coefficient.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, (float, int)):
            raise e.TypeError('`alpha` should be a float or integer')
        if alpha < 0:
            raise e.ValueError('`alpha` should be >= 0')

        self._alpha = alpha

    @property
    def delta(self):
        """float: Second exploitation adjustment coefficient.

        """

        return self._delta

    @delta.setter
    def delta(self, delta):
        if not isinstance(delta, (float, int)):
            raise e.TypeError('`delta` should be a float or integer')
        if delta < 0:
            raise e.ValueError('`delta` should be >= 0')

        self._delta = delta

    @property
    def n_cycles(self):
        """int: Number of cycles.

        """

        return self._n_cycles

    @n_cycles.setter
    def n_cycles(self, n_cycles):
        if not isinstance(n_cycles, int):
            raise e.TypeError('`n_cycles` should be an integer')
        if n_cycles <= 0:
            raise e.ValueError('`n_cycles` should be > 0')

        self._n_cycles = n_cycles

    @property
    def U(self):
        """float: Cycle regularizer.

        """

        return self._U

    @U.setter
    def U(self, U):
        if not isinstance(U, (float, int)):
            raise e.TypeError('`U` should be a float or integer')
        if U < 0:
            raise e.ValueError('`U` should be >= 0')

        self._U = U

    @property
    def w(self):
        """float: Angle regularizer.

        """

        return self._w

    @w.setter
    def w(self, w):
        if not isinstance(w, (float, int)):
            raise e.TypeError('`w` should be a float or integer')
        if w < 0:
            raise e.ValueError('`w` should be >= 0')

        self._w = w

    def _update(self, agents, best_agent, function, iteration, n_iterations):
        """Method that wraps Aquila Optimizer over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.
            iteration (int): Current iteration value.
            n_iterations (int): Maximum number of iterations.

        """

        # Calculates the mean position of space
        average = np.mean([agent.position for agent in agents], axis=0)

        # Iterates through all agents
        for agent in agents:
            # Makes a deepcopy of current agent
            a = copy.deepcopy(agent)

            # Generates a random number
            r1 = r.generate_uniform_random_number()

            # If current iteration is smaller than 2/3 of maximum iterations
            if iteration <= ((2 / 3) * n_iterations):
                # Generates another random number
                r2 = r.generate_uniform_random_number()

                # If random number is smaller or equal to 0.5
                if r1 <= 0.5:
                    # Updates temporary agent's position (Eq. 3)
                    a.position = best_agent.position * (1 - (iteration / n_iterations)) + \
                        (average - best_agent.position * r2)

                # If random number is bigger than 0.5
                else:
                    # Generates a Lévy distirbution and a random integer
                    levy = d.generate_levy_distribution(size=(agent.n_variables, agent.n_dimensions))
                    idx = r.generate_integer_random_number(high=len(agents))

                    # Creates an evenly-space array of `n_variables`
                    # Also broadcasts it to correct `n_dimensions` size
                    D = np.linspace(1, agent.n_variables, agent.n_variables)
                    D = np.repeat(np.expand_dims(D, -1), agent.n_dimensions, axis=1)

                    # Calculates current cycle value (Eq. 10)
                    cycle = self.n_cycles + self.U * D

                    # Calculates `theta` (Eq. 11)
                    theta = -self.w * D + (3 * np.pi) / 2

                    # Calculates `x` and `y` positioning (Eq. 8 and 9)
                    x = cycle * np.sin(theta)
                    y = cycle * np.cos(theta)

                    # Updates temporary agent's position (Eq. 5)
                    a.position = best_agent.position * levy + agents[idx].position + (y - x) * r2

            # If current iteration is bigger than 2/3 of maximum iterations
            else:
                # Generates another random number
                r2 = r.generate_uniform_random_number()

                # If random number is smaller or equal to 0.5
                if r1 <= 0.5:
                    # Expands both lower and upper bound dimensions
                    lb = np.expand_dims(agent.lb, -1)
                    ub = np.expand_dims(agent.ub, -1)

                    # Updates temporary agent's position (Eq. 13)
                    a.position = (best_agent.position - average) * \
                        self.alpha - r2 + ((ub - lb) * r2 + lb) * self.delta

                # If random number is bigger than 0.5
                else:
                    # Calculates both motions (Eq. 16 and 17)
                    G1 = 2 * r2 - 1
                    G2 = 2 * (1 - (iteration / n_iterations))

                    # Calculates quality function (Eq. 15)
                    QF = iteration ** (G1 / (1 - n_iterations) ** 2)

                    # Generates a Lévy distribution
                    levy = d.generate_levy_distribution(size=(agent.n_variables, agent.n_dimensions))

                    # Updates temporary agent's position (Eq. 14)
                    a.position = QF * best_agent.position - \
                        (G1 * a.position * r2) - G2 * levy + r2 * G1

            # Check agent limits
            a.clip_by_bound()

            # Calculates the fitness for the temporary position
            a.fit = function(a.position)

            # If new fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Copy its position and fitness to the agent
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

    def run(self, space, function, store_best_only=False, pre_evaluate=None):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (bool): If True, only the best agent of each iteration is stored in History.
            pre_evaluate (callable): This function is executed before evaluating the function being optimized.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Initial search space evaluation
        self._evaluate(space, function, hook=pre_evaluate)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.to_file(f'Iteration {t+1}/{space.n_iterations}')

                # Updating agents
                self._update(space.agents, space.best_agent,
                             function, t, space.n_iterations)

                # Checking if agents meet the bounds limits
                space.clip_by_bound()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, hook=pre_evaluate)

                # Every iteration, we need to dump agents and best agent
                history.dump(agents=space.agents, best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.to_file(f'Fitness: {space.best_agent.fit}')
                logger.to_file(f'Position: {space.best_agent.position}')

        return history

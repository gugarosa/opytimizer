"""Grey Wolf Optimizer.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class GWO(Optimizer):
    """A GWO class, inherited from Optimizer.

    This is the designed class to define GWO-related
    variables and methods.

    References:
        S. Mirjalili, S. Mirjalili and A. Lewis. Grey Wolf Optimizer.
        Advances in Engineering Software (2014).

    """

    def __init__(self, algorithm='GWO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> GWO.')

        # Override its parent class with the receiving hyperparams
        super(GWO, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    def _calculate_coefficients(self, a):
        """Calculates the mathematical coefficients.

        Args:
            a (float): Linear constant.

        Returns:
            Both `A` and `C` coefficients.

        """

        # Generates two uniform random numbers
        r1 = r.generate_uniform_random_number()
        r2 = r.generate_uniform_random_number()

        # Calculates the `A` coefficient (Eq. 3.3)
        A = 2 * a * r1 - a

        # Calculates the `C` coefficient (Eq. 3.4)
        C = 2 * r2

        return A, C

    def _update(self, agents, function, iteration, n_iterations):
        """Method that wraps the Grey Wolf Optimization over all agents and variables.

        Args:
            agents (list): List of agents.
            function (Function): A function object.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Sorting agents
        agents.sort(key=lambda x: x.fit)

        # Gathers the best three wolves
        alpha, beta, delta = copy.deepcopy(agents[:3])

        # Defines the linear constant
        a = 2 - 2 * iteration / (n_iterations - 1)

        # Iterates through all agents
        for agent in agents:
            # Makes a deepcopy of current agent
            X = copy.deepcopy(agent)

            # Calculates all coefficients
            A_1, C_1 = self._calculate_coefficients(a)
            A_2, C_2 = self._calculate_coefficients(a)
            A_3, C_3 = self._calculate_coefficients(a)

            # Simulates hunting behavior (Eqs. 3.5 and 3.6)
            X_1 = alpha.position - A_1 * np.fabs(C_1 * alpha.position - agent.position)
            X_2 = beta.position - A_2 * np.fabs(C_2 * beta.position - agent.position)
            X_3 = delta.position - A_3 * np.fabs(C_3 * delta.position - agent.position)

            # Calculates the temporary agent (Eq. 3.7)
            X.position = (X_1 + X_2 + X_3) / 3

            # Clips temporary agent's limits
            X.clip_limits()

            # Evaluates temporary agent's new position
            X.fit = function(X.position)

            # Checks if new fitness is better than current agent's fitness
            if X.fit < agent.fit:
                # Updates the corresponding agent's position
                agent.position = copy.deepcopy(X.position)

                # And its fitness as well
                agent.fit = copy.deepcopy(X.fit)

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
                self._update(space.agents, function, t, space.n_iterations)

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

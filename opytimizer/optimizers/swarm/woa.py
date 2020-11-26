"""Whale Optimization Algorithm.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as log
from opytimizer.core.optimizer import Optimizer

logger = log.get_logger(__name__)


class WOA(Optimizer):
    """A WOA class, inherited from Optimizer.

    This is the designed class to define WOA-related
    variables and methods.

    References:
        S. Mirjalli and A. Lewis. The Whale Optimization Algorithm.
        Advances in Engineering Software (2016).

    """

    def __init__(self, algorithm='WOA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Override its parent class with the receiving hyperparams
        super(WOA, self).__init__(algorithm)

        # Logarithmic spiral
        self.b = 1

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def b(self):
        """float: Logarithmic spiral.

        """

        return self._b

    @b.setter
    def b(self, b):
        if not isinstance(b, (float, int)):
            raise e.TypeError('`b` should be a float or integer')

        self._b = b

    def _generate_random_agent(self, agent):
        """Generates a new random-based agent.

        Args:
            agent (Agent): Agent to be copied.

        Returns:
            Random-based agent.

        """

        # Makes a deepcopy of agent
        a = copy.deepcopy(agent)

        # Iterates through all decision variables
        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            # For each decision variable, we generate uniform random numbers
            a.position[j] = r.generate_uniform_random_number(
                lb, ub, a.n_dimensions)

        return a

    def _update(self, agents, best_agent, coefficient):
        """Method that wraps Whale Optimization Algorithm updates.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            coefficient (float): A linearly decreased coefficient.

        """

        # Iterates through all agents
        for agent in agents:
            # Generating an uniform random number
            r1 = r.generate_uniform_random_number()

            # Calculates the `A` coefficient
            A = 2 * coefficient * r1 - coefficient

            # Calculates the `C` coefficient
            C = 2 * r1

            # Generates a random number between 0 and 1
            p = r.generate_uniform_random_number()

            # If `p` is smaller than 0.5
            if p < 0.5:
                # If `A` is smaller than 1
                if np.fabs(A) < 1:
                    # Calculates the distance coefficient
                    D = np.fabs(C * best_agent.position - agent.position)

                    # Updates the agent's position
                    agent.position = best_agent.position - A * D

                # If `A` is bigger or equal to 1
                else:
                    # Generates a random-based agent
                    a = self._generate_random_agent(agent)

                    # Calculates the distance coefficient
                    D = np.fabs(C * a.position - agent.position)

                    # Updates the agent's position
                    agent.position = a.position - A * D

            # If `p` is bigger or equal to 1
            else:
                # Generates a random number between -1 and 1
                l = r.generate_gaussian_random_number()

                # Calculates the distance coefficient
                D = np.fabs(best_agent.position - agent.position)

                # Updates the agent's position
                agent.position = D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best_agent.position

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

                # Linearly decreases the coefficient
                a = 2 - 2 * t / (space.n_iterations - 1)

                # Updating agents
                self._update(space.agents, space.best_agent, a)

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

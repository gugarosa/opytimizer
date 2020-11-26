"""Moth-Flame Optimization.
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


class MFO(Optimizer):
    """A MFO class, inherited from Optimizer.

    This is the designed class to define MFO-related
    variables and methods.

    References:
        S. Mirjalili. Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm.
        Knowledge-Based Systems (2015).

    """

    def __init__(self, algorithm='MFO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> MFO.')

        # Override its parent class with the receiving hyperparams
        super(MFO, self).__init__(algorithm)

        # Spiral constant
        self.b = 1

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def b(self):
        """float: Spiral constant.

        """

        return self._b

    @b.setter
    def b(self, b):
        if not isinstance(b, (float, int)):
            raise e.TypeError('`b` should be a float or integer')
        if b < 0:
            raise e.ValueError('`b` should be >= 0')

        self._b = b

    def _update(self, agents, iteration, n_iterations):
        """Method that wraps global and local pollination updates over all agents and variables.

        Args:
            agents (list): List of agents.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Makes a deepcopy of current population
        flames = copy.deepcopy(agents)

        # Sorts the flames
        flames.sort(key=lambda x: x.fit)

        # Calculates the number of flames (eq. 3.14)
        n_flames = int(len(flames) - iteration * ((len(flames) - 1) / n_iterations)) - 1

        # Calculates the convergence constant
        r = -1 + iteration * (-1 / n_iterations)

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Iterates through every decision variable
            for j in range(agent.n_variables):
                # Generates a random `t`
                t = rnd.generate_uniform_random_number(r, 1)

                # Checks if current moth should be updated with corresponding flame
                if i < n_flames:
                    # Calculates the distance (eq. 3.13)
                    D = np.fabs(flames[i].position[j] - agent.position[j])

                    # Updates current agent's position (eq. 3.12)
                    agent.position[j] = D * np.exp(self.b * t) * \
                        np.cos(2 * np.pi * t) + flames[i].position[j]

                # If current moth should be updated with best flame
                else:
                    # Calculates the distance (eq. 3.13)
                    D = np.fabs(flames[0].position[j] - agent.position[j])

                    # Updates current agent's position (eq. 3.12)
                    agent.position[j] = D * np.exp(self.b * t) * \
                        np.cos(2 * np.pi * t) + flames[0].position[j]

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
                self._update(space.agents, t, space.n_iterations)

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

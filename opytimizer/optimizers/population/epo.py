"""Emperor Penguin Optimizer.
"""

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as log
from opytimizer.core.optimizer import Optimizer

logger = log.get_logger(__name__)


class EPO(Optimizer):
    """An EPO class, inherited from Optimizer.

    This is the designed class to define EPO-related
    variables and methods.

    References:
        G. Dhiman and V. Kumar. Emperor penguin optimizer: A bio-inspired algorithm for engineering problems.
        Knowledge-Based Systems (2018).

    """

    def __init__(self, algorithm='EPO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> EPO.')

        # Override its parent class with the receiving hyperparams
        super(EPO, self).__init__(algorithm)

        # Exploration control parameter
        self.f = 2.0

        # Exploitation control parameter
        self.l = 1.5

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def f(self):
        """float: Exploration control parameter.

        """

        return self._f

    @f.setter
    def f(self, f):
        if not isinstance(f, (float, int)):
            raise e.TypeError('`f` should be a float or integer')

        self._f = f

    @property
    def l(self):
        """float: Exploitation control parameter.

        """

        return self._l

    @l.setter
    def l(self, l):
        if not isinstance(l, (float, int)):
            raise e.TypeError('`l` should be a float or integer')

        self._l = l

    def _update(self, agents, best_agent, iteration, n_iterations):
        """Method that wraps the Emperor Penguin Optimization over all agents and variables.
        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Iterates through every agent
        for agent in agents:
            # Generates a radius constant
            R = r.generate_uniform_random_number()

            # Checks if radius is bigger or equal to 0.5
            if R >= 0.5:
                # Defines temperature as zero
                T = 0

            # If radius is smaller than one
            else:
                # Defines temperature as one
                T = 1

            # Calculates the temperature profile (Eq. 7)
            T_p = T - n_iterations / (iteration - n_iterations)

            # Calculates the polygon grid accuracy (Eq. 10)
            P_grid = np.fabs(best_agent.position - agent.position)

            # Generates a uniform random number and the `C` coefficient
            r1 = r.generate_uniform_random_number()
            C = r.generate_uniform_random_number(size=agent.n_variables)

            # Calculates the avoidance coefficient (Eq. 9)
            A = 2 * (T_p + P_grid) * r1 - T_p

            # Calculates the social forces of emperor penguin (Eq. 12)
            S = (np.fabs(self.f * np.exp(-iteration / self.l) - np.exp(-iteration))) ** 2

            # Calculates the distance between current agent and emperor penguin (Eq. 8)
            D_ep = np.fabs(S * best_agent.position - C * agent.position)

            # Updates current agent's position (Eq. 13)
            agent.position = best_agent.position - A * D_ep

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
                self._update(space.agents, space.best_agent, t, space.n_iterations)

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

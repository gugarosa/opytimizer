"""Arithmetic Optimization Algorithm.
"""

from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class AOA(Optimizer):
    """An AOA class, inherited from Optimizer.

    This is the designed class to define AOA-related
    variables and methods.

    References:
        L. Abualigah et al. The Arithmetic Optimization Algorithm.
        Computer Methods in Applied Mechanics and Engineering (2021).

    """

    def __init__(self, algorithm='AOA', params=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> AOA.')

        # Override its parent class with the receiving params
        super(AOA, self).__init__(algorithm)

        # Minimum accelerated function
        self.a_min = 0.2

        # Maximum accelerated function
        self.a_max = 1.0

        # Sensitive parameter
        self.alpha = 5.0

        # Control parameter
        self.mu = 0.499

        # Now, we need to build this class up
        self._build(params)

        logger.info('Class overrided.')

    @property
    def a_min(self):
        """float: Minimum accelerated function.

        """

        return self._a_min

    @a_min.setter
    def a_min(self, a_min):
        if not isinstance(a_min, (float, int)):
            raise e.TypeError('`a_min` should be a float or integer')
        if a_min < 0:
            raise e.ValueError('`a_min` should be >= 0')

        self._a_min = a_min

    @property
    def a_max(self):
        """float: Maximum accelerated function.

        """

        return self._a_max

    @a_max.setter
    def a_max(self, a_max):
        if not isinstance(a_max, (float, int)):
            raise e.TypeError('`a_max` should be a float or integer')
        if a_max < 0:
            raise e.ValueError('`a_max` should be >= 0')
        if a_max < self.a_min:
            raise e.ValueError('`a_max` should be >= `a_min`')

        self._a_max = a_max

    @property
    def alpha(self):
        """float: Sensitive parameter.

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
    def mu(self):
        """float: Control parameter.

        """

        return self._mu

    @mu.setter
    def mu(self, mu):
        if not isinstance(mu, (float, int)):
            raise e.TypeError('`mu` should be a float or integer')
        if mu < 0:
            raise e.ValueError('`mu` should be >= 0')

        self._mu = mu

    def _update(self, agents, best_agent, iteration, n_iterations):
        """Method that wraps Arithmetic Optimization Algorithm over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            iteration (int): Current iteration value.
            n_iterations (int): Maximum number of iterations.

        """

        # Calculating math optimizer accelarated coefficient (eq. 2)
        MOA = self.a_min + iteration * ((self.a_max - self.a_min) / n_iterations)

        # Calculating math optimizer probability (eq. 4)
        MOP = 1 - (iteration ** (1 / self.alpha) / n_iterations ** (1 / self.alpha))

        # Iterates through all agents
        for agent in agents:
            # Iterates through all variables
            for j in range(agent.n_variables):
                # Generating random probability
                r1 = r.generate_uniform_random_number()

                # Calculates the search partition
                search_partition = (agent.ub[j] - agent.lb[j]) * self.mu + agent.lb[j]

                # If probability is bigger than MOA
                if r1 > MOA:
                    # Generates an extra probability
                    r2 = r.generate_uniform_random_number()

                    # If probability is bigger than 0.5
                    if r2 > 0.5:
                        # Updates position with (eq. 3 - top)
                        agent.position[j] = best_agent.position[j] / (MOP + c.EPSILON) * search_partition

                    # If probability is smaller than 0.5
                    else:
                        # Updates position with (eq. 3 - bottom)
                        agent.position[j] = best_agent.position[j] * MOP * search_partition

                # If probability is smaller than MOA
                else:
                    # Generates an extra probability
                    r3 = r.generate_uniform_random_number()

                    # If probability is bigger than 0.5
                    if r3 > 0.5:
                        # Updates position with (eq. 5 - top)
                        agent.position[j] = best_agent.position[j] - MOP * search_partition

                    # If probability is smaller than 0.5
                    else:
                        # Updates position with (eq. 5 - bottom)
                        agent.position[j] = best_agent.position[j] + MOP * search_partition

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
                self._update(space.agents, space.best_agent, t, space.n_iterations)

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

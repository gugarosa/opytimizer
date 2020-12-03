"""Tug Of War Optimization.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class TWO(Optimizer):
    """A TWO class, inherited from Optimizer.

    This is the designed class to define TWO-related
    variables and methods.

    References:
        A. Kaveh. Tug of War Optimization.
        Advances in Metaheuristic Algorithms for Optimal Design of Structures (2016).

    """

    def __init__(self, algorithm='TWO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> TWO.')

        # Override its parent class with the receiving hyperparams
        super(TWO, self).__init__(algorithm)

        # Static friction coefficient
        self.mu_s = 1

        # Kinematic friction coefficient
        self.mu_k = 1

        # Time displacement
        self.delta_t = 1

        # Speed constant
        self.alpha = 0.9

        # Scaling factor
        self.beta = 0.05

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def mu_s(self):
        """float: Static friction coefficient.

        """

        return self._mu_s

    @mu_s.setter
    def mu_s(self, mu_s):
        if not isinstance(mu_s, (float, int)):
            raise e.TypeError('`mu_s` should be a float or integer')
        if mu_s < 0:
            raise e.ValueError('`mu_s` should be >= 0')

        self._mu_s = mu_s

    @property
    def mu_k(self):
        """float: Kinematic friction coefficient.

        """

        return self._mu_k

    @mu_k.setter
    def mu_k(self, mu_k):
        if not isinstance(mu_k, (float, int)):
            raise e.TypeError('`mu_k` should be a float or integer')
        if mu_k < 0:
            raise e.ValueError('`mu_k` should be >= 0')

        self._mu_k = mu_k

    @property
    def delta_t(self):
        """float: Time displacement.

        """

        return self._delta_t

    @delta_t.setter
    def delta_t(self, delta_t):
        if not isinstance(delta_t, (float, int)):
            raise e.TypeError('`delta_t` should be a float or integer')
        if delta_t < 0:
            raise e.ValueError('`delta_t` should be >= 0')

        self._delta_t = delta_t

    @property
    def alpha(self):
        """float: Speed constant.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, (float, int)):
            raise e.TypeError('`alpha` should be a float or integer')
        if alpha < 0.9 or alpha > 1:
            raise e.ValueError('`alpha` should be between 0.9 and 1')

        self._alpha = alpha

    @property
    def beta(self):
        """float: Scaling factor.

        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        if not isinstance(beta, (float, int)):
            raise e.TypeError('`beta` should be a float or integer')
        if beta <= 0 or beta > 1:
            raise e.ValueError('`beta` should be greater than 0 and less than 1')

        self._beta = beta

    def _constraint_handle(self, agents, best_agent, function, iteration):
        """Performs the constraint handling procedure (eq. 11).

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.
            iteration (int): Current iteration.

        """

        # Iterates through every agent
        for agent in agents:
            # Generates a random number
            r1 = r.generate_uniform_random_number()

            # If random is smaller than 0.5
            if r1 < 0.5:
                # Generates a gaussian random number
                r2 = r.generate_gaussian_random_number()

                # Updates the agent's position
                agent.position = best_agent.position + \
                    (r2 / iteration) * (best_agent.position - agent.position)

            # Clips its limits
            agent.clip_limits()

            # Re-calculates its fitness
            agent.fit = function(agent.position)

    def _update(self, agents, best_agent, function, iteration, n_iterations):
        """Method that wraps Tug of War Optimization over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Sorting agents
        agents.sort(key=lambda x: x.fit)

        # Gathers best and worst fitness
        best_fit, worst_fit = agents[0].fit, agents[-1].fit

        # Calculates the agents' weights
        weights = [(agent.fit - worst_fit) /
                   (best_fit - worst_fit + c.EPSILON) + 1 for agent in agents]

        # We copy a temporary list for iterating purposes
        temp_agents = copy.deepcopy(agents)

        # Linearly decreasing `mu_k`
        mu_k = self.mu_k - (self.mu_k - 0.1) * (iteration / n_iterations)

        # Iterating through 'i' agents
        for i, temp1 in enumerate(temp_agents):
            # Initializes `delta` as zero
            delta = 0.0

            # Iterating through 'j' agents
            for j, temp2 in enumerate(temp_agents):
                # If weight from agent `i` is smaller than weight from agent `j`
                if weights[i] < weights[j]:
                    # Calculates the residual force (eq. 6)
                    force = np.maximum(weights[i] * self.mu_s, weights[j] * self.mu_s) - weights[i] * mu_k

                    # Calculates the gravitational acceleration (eq. 8)
                    g = temp2.position - temp1.position

                    # Calculates the acceleration (eq. 7)
                    acceleration = (force / (weights[i] * mu_k)) * g

                    # Generates a random gaussian number
                    r1 = r.generate_gaussian_random_number(size=(temp1.n_variables, temp1.n_dimensions))

                    # Calculates the displacement (eq. 9-10)
                    delta += 0.5 * acceleration * self.delta_t ** 2 + np.multiply(self.alpha ** iteration * self.beta * (
                        np.expand_dims(temp1.ub, -1) - np.expand_dims(temp1.lb, -1)), r1)

            # Updates the temporary agent's position (eq. 11)
            temp1.position += delta

        # Performs the constraint handling
        self._constraint_handle(temp_agents, best_agent, function, iteration)

        # Iterates through real and temporary populations
        for agent, temp in zip(agents, temp_agents):
            # If temporary agent is better than real one
            if temp.fit < agent.fit:
                # Updates its position
                agent.position = copy.deepcopy(temp.position)

                # And updates its fitness
                agent.fit = copy.deepcopy(temp.fit)

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
                self._update(space.agents, space.best_agent, function, t+1, space.n_iterations)

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

"""Artificial Butterfly Optimization.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class ABO(Optimizer):
    """An ABO class, inherited from Optimizer.

    This is the designed class to define ABO-related
    variables and methods.

    References:
        X. Qi, Y. Zhu and H. Zhang. A new meta-heuristic butterfly-inspired algorithm.
        Journal of Computational Science (2017).

    """

    def __init__(self, algorithm='ABO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> ABO.')

        # Override its parent class with the receiving hyperparams
        super(ABO, self).__init__(algorithm)

        # Ratio of sunspot butterflies
        self.sunspot_ratio = 0.9

        # Free flight constant
        self.a = 2.0

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def sunspot_ratio(self):
        """float: Ratio of sunspot butterflies.

        """

        return self._sunspot_ratio

    @sunspot_ratio.setter
    def sunspot_ratio(self, sunspot_ratio):
        if not isinstance(sunspot_ratio, (float, int)):
            raise e.TypeError('`sunspot_ratio` should be a float or integer')
        if sunspot_ratio < 0 or sunspot_ratio > 1:
            raise e.ValueError('`sunspot_ratio` should be between 0 and 1')

        self._sunspot_ratio = sunspot_ratio

    @property
    def a(self):
        """float: Free flight constant.

        """

        return self._a

    @a.setter
    def a(self, a):
        if not isinstance(a, (float, int)):
            raise e.TypeError('`a` should be a float or integer')
        if a < 0:
            raise e.ValueError('`a` should be >= 0')

        self._a = a

    def _flight_mode(self, agent, neighbour, function):
        """Flies to a new location according to the flight mode (eq. 1).

        Args:
            agent (Agent): Current agent.
            neighbour (Agent): Selected neigbour.
            function (Function): A Function object that will be used as the objective function.

        Returns:
            Current agent or an agent with updated position, along with a boolean that indicates whether
            agent is better or not than current one.

        """

        # Generates a random decision variable index
        j = r.generate_integer_random_number(0, agent.n_variables)

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number(-1, 1)

        # Makes a deepcopy of current agent
        temp = copy.deepcopy(agent)

        # Updates temporary agent's position (eq. 1)
        temp.position[j] = agent.position[j] + (agent.position[j] - neighbour.position[j]) * r1

        # Clips its limits
        temp.clip_limits()

        # Re-calculates its fitness
        temp.fit = function(temp.position)

        # If its fitness is better than current agent
        if temp.fit < agent.fit:
            # Return temporary agent as well as a true variable
            return temp.position, temp.fit, True

        # Return current agent as well as a false variable
        return agent.position, agent.fit, False

    def _update(self, agents, function, iteration, n_iterations):
        """Method that wraps Artificial Butterfly Optimization over all agents and variables.

        Args:
            agents (list): List of agents.
            function (Function): A Function object that will be used as the objective function.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Sorting agents
        agents.sort(key=lambda x: x.fit)

        # Calculates the number of sunspot butterflies
        n_sunspots = int(self.sunspot_ratio * len(agents))

        # Iterates through all sunspot butterflies
        for agent in agents[:n_sunspots]:
            # Generates the index for a random sunspot butterfly
            k = r.generate_integer_random_number(0, len(agents))

            # Performs a flight mode using sunspot butterflies (eq. 1)
            agent.position, agent.fit, _ = self._flight_mode(agent, agents[k], function)

        # Iterates through all canopy butterflies
        for agent in agents[n_sunspots:]:
            # Generates the index for a random canopy butterfly
            k = r.generate_integer_random_number(0, len(agents) - n_sunspots)

            # Performs a flight mode using canopy butterflies (eq. 1)
            agent.position, agent.fit, is_better = self._flight_mode(agent, agents[k], function)

            # If there was not fitness replacement
            if not is_better:
                # Generates the index for a random butterfly
                k = r.generate_integer_random_number(0, len(agents))

                # Generates random uniform number
                r1 = r.generate_uniform_random_number()

                # Calculates `D` (eq. 4)
                D = np.fabs(2 * r1 * agents[k].position - agent.position)

                # Generates another random uniform number
                r2 = r.generate_uniform_random_number()

                # Linearly decreases `a`
                a = (self.a - self.a * (iteration / n_iterations))

                # Updates the agent's position (eq. 3)
                agent.position = agents[k].position - 2 * a * r2 - a * D

                # Clips its limits
                agent.clip_limits()

                # Re-calculates its fitness
                agent.fit = function(agent.position)

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

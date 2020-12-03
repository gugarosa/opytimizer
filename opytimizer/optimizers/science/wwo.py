"""Water Wave Optimization.
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


class WWO(Optimizer):
    """A WWO class, inherited from Optimizer.

    This is the designed class to define WWO-related
    variables and methods.

    References:
        Y.-J. Zheng. Water wave optimization: A new nature-inspired metaheuristic.
        Computers & Operations Research (2015).

    """

    def __init__(self, algorithm='WWO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> WWO.')

        # Override its parent class with the receiving hyperparams
        super(WWO, self).__init__(algorithm)

        # Maximum wave height
        self.h_max = 5

        # Wave length reduction coefficient
        self.alpha = 1.001

        # Breaking coefficient
        self.beta = 0.001

        # Maximum number of breakings
        self.k_max = 1

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def h_max(self):
        """int: Maximum wave height.

        """

        return self._h_max

    @h_max.setter
    def h_max(self, h_max):
        if not isinstance(h_max, int):
            raise e.TypeError('`h_max` should be an integer')
        if h_max <= 0:
            raise e.ValueError('`h_max` should be > 0')

        self._h_max = h_max

    @property
    def alpha(self):
        """float: Wave length reduction coefficient.

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
    def beta(self):
        """float: Breaking coefficient.

        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        if not isinstance(beta, (float, int)):
            raise e.TypeError('`beta` should be a float or integer')
        if beta < 0:
            raise e.ValueError('`beta` should be >= 0')

        self._beta = beta

    @property
    def k_max(self):
        """int: Maximum number of breakings.

        """

        return self._k_max

    @k_max.setter
    def k_max(self, k_max):
        if not isinstance(k_max, int):
            raise e.TypeError('`k_max` should be an integer')
        if k_max <= 0:
            raise e.ValueError('`k_max` should be > 0')

        self._k_max = k_max

    def _propagate_wave(self, agent, function, length):
        """Propagates wave into a new position (eq. 6).

        Args:
            function (Function): A function object.
            length (np.array): Array of wave lengths.

        Returns:
            Propagated wave.

        """

        # Makes a deepcopy of current agent
        wave = copy.deepcopy(agent)

        # Iterates through all variables
        for j in range(wave.n_variables):
            # Generates a uniform random number
            r1 = r.generate_uniform_random_number(-1, 1)

            # Updates the wave's position
            wave.position[j] += r1 * length * (j + 1)

        # Clips its limits
        wave.clip_limits()

        # Re-calculates its fitness
        wave.fit = function(wave.position)

        return wave

    def _refract_wave(self, agent, best_agent, function, length):
        """Refract wave into a new position (eq. 8-9).

        Args:
            agent (Agent): Agent to be refracted.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            length (np.array): Array of wave lengths.

        Returns:
            New height and length values.

        """

        # Gathers current fitness
        current_fit = agent.fit

        # Iterates through all variables
        for j in range(agent.n_variables):
            # Calculates a mean value
            mean = (best_agent.position[j] + agent.position[j]) / 2

            # Calculates the standard deviation
            std = np.fabs(best_agent.position[j] - agent.position[j]) / 2

            # Generates a new position (Eq. 8)
            agent.position[j] = r.generate_gaussian_random_number(mean, std)

        # Clips its limits
        agent.clip_limits()

        # Re-calculates its fitness
        agent.fit = function(agent.position)

        # Updates the new height to maximum height value
        new_height = self.h_max

        # Re-calculates the new length (Eq. 9)
        new_length = length * (current_fit / (agent.fit + c.EPSILON))

        return new_height, new_length

    def _break_wave(self, wave, function, j):
        """Breaks current wave into a new one (eq. 10).

        Args:
            wave (Agent): Wave to be broken.
            function (Function): A function object.
            j (int): Index of dimension to be broken.

        Returns:
            Broken wave.

        """

        # Makes a deepcopy of current wave
        broken_wave = copy.deepcopy(wave)

        # Generates a gaussian random number
        r1 = r.generate_gaussian_random_number()

        # Updates the broken wave's position
        broken_wave.position[j] += r1 * self.beta * (j + 1)

        # Clips its limits
        broken_wave.clip_limits()

        # Re-calculates its fitness
        broken_wave.fit = function(broken_wave.position)

        return broken_wave

    def _update_wave_length(self, agents, length):
        """Updates the wave length of current population.

        Args:
            agents (list): List of agents.
            length (np.array): Array of wave lengths.

        """

        # Sorts the agents
        agents.sort(key=lambda x: x.fit)

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Updates its length
            length[i] *= self.alpha ** -((agent.fit - agents[-1].fit + c.EPSILON) / (
                agents[0].fit - agents[-1].fit + c.EPSILON))

    def _update(self, agents, best_agent, function, height, length):
        """Method that wraps Water Wave Optimization over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            height (np.array): Array of wave heights.
            length (np.array): Array of wave lengths.

        """

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Propagates a wave into a new temporary one (Eq. 6)
            wave = self._propagate_wave(agent, function, length[i])

            # Checks if propagated wave is better than current one
            if wave.fit < agent.fit:
                # Also checks if propagated wave is better than global one
                if wave.fit < best_agent.fit:
                    # Replaces the best agent with propagated wave
                    best_agent.position = copy.deepcopy(wave.position)
                    best_agent.fit = copy.deepcopy(wave.fit)

                    # Generates a `k` number of breaks
                    k = r.generate_integer_random_number(1, self.k_max + 1)

                    # Iterates through every possible break
                    for j in range(k):
                        # Breaks the propagated wave (Eq. 10)
                        broken_wave = self._break_wave(wave, function, j)

                        # Checks if broken wave is better than global one
                        if broken_wave.fit < best_agent.fit:
                            # Replaces the best agent with broken wave
                            best_agent.position = copy.deepcopy(broken_wave.position)
                            best_agent.fit = copy.deepcopy(broken_wave.fit)

                # Replaces current agent's with propagated wave
                agent.position = copy.deepcopy(wave.position)
                agent.fit = copy.deepcopy(wave.fit)

                # Sets its height to maximum height
                height[i] = self.h_max

            # If propagated wave is not better than current agent
            else:
                # Decreases its height by one
                height[i] -= 1

                # If its height reaches zero
                if height[i] == 0:
                    # Refracts the wave and generates a new height and wave length (Eq. 8-9)
                    height[i], length[i] = self._refract_wave(agent, best_agent, function, length[i])

        # Updates the wave length for all agents (Eq. 7)
        self._update_wave_length(agents, length)

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

        # Creates a height vector with `h_max` values
        height = r.generate_uniform_random_number(self.h_max, self.h_max, space.n_agents)

        # Creates a length vector with 0.5 values
        length = r.generate_uniform_random_number(0.5, 0.5, space.n_agents)

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
                self._update(space.agents, space.best_agent, function, height, length)

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

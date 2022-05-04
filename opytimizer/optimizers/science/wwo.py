"""Water Wave Optimization.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class WWO(Optimizer):
    """A WWO class, inherited from Optimizer.

    This is the designed class to define WWO-related
    variables and methods.

    References:
        Y.-J. Zheng. Water wave optimization: A new nature-inspired metaheuristic.
        Computers & Operations Research (2015).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> WWO.")

        # Overrides its parent class with the receiving params
        super(WWO, self).__init__()

        # Maximum wave height
        self.h_max = 5

        # Wave length reduction coefficient
        self.alpha = 1.001

        # Breaking coefficient
        self.beta = 0.001

        # Maximum number of breakings
        self.k_max = 1

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def h_max(self) -> int:
        """Maximum wave height."""

        return self._h_max

    @h_max.setter
    def h_max(self, h_max: int) -> None:
        if not isinstance(h_max, int):
            raise e.TypeError("`h_max` should be an integer")
        if h_max <= 0:
            raise e.ValueError("`h_max` should be > 0")

        self._h_max = h_max

    @property
    def alpha(self) -> float:
        """Wave length reduction coefficient."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        if alpha < 0:
            raise e.ValueError("`alpha` should be >= 0")

        self._alpha = alpha

    @property
    def beta(self) -> float:
        """Breaking coefficient."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        if beta < 0:
            raise e.ValueError("`beta` should be >= 0")

        self._beta = beta

    @property
    def k_max(self) -> int:
        """Maximum number of breakings."""

        return self._k_max

    @k_max.setter
    def k_max(self, k_max: int) -> None:
        if not isinstance(k_max, int):
            raise e.TypeError("`k_max` should be an integer")
        if k_max <= 0:
            raise e.ValueError("`k_max` should be > 0")

        self._k_max = k_max

    @property
    def height(self) -> np.ndarray:
        """Array of heights."""

        return self._height

    @height.setter
    def height(self, height: np.ndarray) -> None:
        if not isinstance(height, np.ndarray):
            raise e.TypeError("`height` should be a numpy array")

        self._height = height

    @property
    def length(self) -> np.ndarray:
        """Array of lengths."""

        return self._length

    @length.setter
    def length(self, length: np.ndarray) -> None:
        if not isinstance(length, np.ndarray):
            raise e.TypeError("`length` should be a numpy array")

        self._length = length

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Arrays of heights and lengths
        self.height = r.generate_uniform_random_number(
            self.h_max, self.h_max, space.n_agents
        )
        self.length = r.generate_uniform_random_number(0.5, 0.5, space.n_agents)

    def _propagate_wave(self, agent: Agent, function: Function, index: int) -> Agent:
        """Propagates wave into a new position (eq. 6).

        Args:
            agent: Current wave.
            function: A function object.
            index: Index of wave length.

        Returns:
            (Agent): Propagated wave.

        """

        # Makes a deep copy of current agent
        wave = copy.deepcopy(agent)

        # Iterates through all variables
        for j in range(wave.n_variables):
            # Generates a uniform random number
            r1 = r.generate_uniform_random_number(-1, 1)

            # Updates the wave's position
            wave.position[j] += r1 * self.length[index] * (j + 1)

        # Clips its limits
        wave.clip_by_bound()

        # Re-calculates its fitness
        wave.fit = function(wave.position)

        return wave

    def _refract_wave(
        self, agent: Agent, best_agent: Agent, function: Function, index: int
    ) -> Tuple[float, float]:
        """Refract wave into a new position (eq. 8-9).

        Args:
            agent: Agent to be refracted.
            best_agent: Global best agent.
            function: A function object.
            index: Index of wave length.

        Returns:
            (Tuple[float, float]): New height and length values.

        """

        # Gathers current fitness
        current_fit = agent.fit

        # Iterates through all variables
        for j in range(agent.n_variables):
            # Calculates a mean value
            mean = (best_agent.position[j] + agent.position[j]) / 2

            # Calculates the standard deviation
            std = np.fabs(best_agent.position[j] - agent.position[j]) / 2

            # Generates a new position (eq. 8)
            agent.position[j] = r.generate_gaussian_random_number(mean, std)

        # Clips its limits
        agent.clip_by_bound()

        # Re-calculates its fitness
        agent.fit = function(agent.position)

        # Updates the new height to maximum height value
        new_height = self.h_max

        # Re-calculates the new length (eq. 9)
        new_length = self.length[index] * (current_fit / (agent.fit + c.EPSILON))

        return new_height, new_length

    def _break_wave(self, wave: Agent, function: Function, j: int) -> Agent:
        """Breaks current wave into a new one (eq. 10).

        Args:
            wave: Wave to be broken.
            function: A function object.
            j: Index of dimension to be broken.

        Returns:
            (Agent): Broken wave.

        """

        # Makes a deep copy of current wave
        broken_wave = copy.deepcopy(wave)

        # Generates a gaussian random number
        r1 = r.generate_gaussian_random_number()

        # Updates the broken wave's position
        broken_wave.position[j] += r1 * self.beta * (j + 1)

        # Clips its limits
        broken_wave.clip_by_bound()

        # Re-calculates its fitness
        broken_wave.fit = function(broken_wave.position)

        return broken_wave

    def _update_wave_length(self, agents: List[Agent]) -> None:
        """Updates the wave length of current population.

        Args:
            agents: List of agents.

        """

        # Sorts agents
        agents.sort(key=lambda x: x.fit)

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Updates its length
            self.length[i] *= self.alpha ** -(
                (agent.fit - agents[-1].fit + c.EPSILON)
                / (agents[0].fit - agents[-1].fit + c.EPSILON)
            )

    def update(self, space: Space, function: Function) -> None:
        """Wraps Water Wave Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A function object.

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Propagates a wave into a new temporary one (eq. 6)
            wave = self._propagate_wave(agent, function, i)

            # Checks if propagated wave is better than current one
            if wave.fit < agent.fit:
                # Also checks if propagated wave is better than global one
                if wave.fit < space.best_agent.fit:
                    # Replaces the best agent with propagated wave
                    space.best_agent.position = copy.deepcopy(wave.position)
                    space.best_agent.fit = copy.deepcopy(wave.fit)

                    # Generates a `k` number of breaks
                    k = r.generate_integer_random_number(1, self.k_max + 1)

                    # Iterates through every possible break
                    for j in range(k):
                        # Breaks the propagated wave (eq. 10)
                        broken_wave = self._break_wave(wave, function, j)

                        # Checks if broken wave is better than global one
                        if broken_wave.fit < space.best_agent.fit:
                            # Replaces the best agent with broken wave
                            space.best_agent.position = copy.deepcopy(
                                broken_wave.position
                            )
                            space.best_agent.fit = copy.deepcopy(broken_wave.fit)

                # Replaces current agent's with propagated wave
                agent.position = copy.deepcopy(wave.position)
                agent.fit = copy.deepcopy(wave.fit)

                # Sets its height to maximum height
                self.height[i] = self.h_max

            # If propagated wave is not better than current agent
            else:
                # Decreases its height by one
                self.height[i] -= 1

                # If its height reaches zero
                if self.height[i] == 0:
                    # Refracts the wave and generates a new height and wave length (eq. 8-9)
                    self.height[i], self.length[i] = self._refract_wave(
                        agent, space.best_agent, function, i
                    )

        # Updates the wave length for all agents (eq. 7)
        self._update_wave_length(space.agents)

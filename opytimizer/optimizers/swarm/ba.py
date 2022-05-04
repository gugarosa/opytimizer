"""Bat Algorithm.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as rnd
import opytimizer.utils.exception as ex
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class BA(Optimizer):
    """A BA class, inherited from Optimizer.

    This is the designed class to define BA-related
    variables and methods.

    References:
        X.-S. Yang. A new metaheuristic bat-inspired algorithm.
        Nature inspired cooperative strategies for optimization (2010).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> BA.")

        # Overrides its parent class with the receiving params
        super(BA, self).__init__()

        # Minimum frequency range
        self.f_min = 0

        # Maximum frequency range
        self.f_max = 2

        # Loudness parameter
        self.A = 0.5

        # Pulse rate
        self.r = 0.5

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def f_min(self) -> float:
        """Minimum frequency range."""

        return self._f_min

    @f_min.setter
    def f_min(self, f_min: float) -> None:
        if not isinstance(f_min, (float, int)):
            raise ex.TypeError("`f_min` should be a float or integer")
        if f_min < 0:
            raise ex.ValueError("`f_min` should be >= 0")

        self._f_min = f_min

    @property
    def f_max(self) -> float:
        """Maximum frequency range."""

        return self._f_max

    @f_max.setter
    def f_max(self, f_max: float) -> None:
        if not isinstance(f_max, (float, int)):
            raise ex.TypeError("`f_max` should be a float or integer")
        if f_max < 0:
            raise ex.ValueError("`f_max` should be >= 0")
        if f_max < self.f_min:
            raise ex.ValueError("`f_max` should be >= `f_min`")

        self._f_max = f_max

    @property
    def A(self) -> float:
        """Loudness parameter."""

        return self._A

    @A.setter
    def A(self, A: float) -> None:
        if not isinstance(A, (float, int)):
            raise ex.TypeError("`A` should be a float or integer")
        if A < 0:
            raise ex.ValueError("`A` should be >= 0")

        self._A = A

    @property
    def r(self) -> float:
        """Pulse rate."""

        return self._r

    @r.setter
    def r(self, r: float) -> None:
        if not isinstance(r, (float, int)):
            raise ex.TypeError("`r` should be a float or integer")
        if r < 0:
            raise ex.ValueError("`r` should be >= 0")

        self._r = r

    @property
    def frequency(self) -> np.ndarray:
        """Array of frequencies."""

        return self._frequency

    @frequency.setter
    def frequency(self, frequency: np.ndarray) -> None:
        if not isinstance(frequency, np.ndarray):
            raise ex.TypeError("`frequency` should be a numpy array")

        self._frequency = frequency

    @property
    def velocity(self) -> np.ndarray:
        """Array of velocities."""

        return self._velocity

    @velocity.setter
    def velocity(self, velocity: np.ndarray) -> None:
        if not isinstance(velocity, np.ndarray):
            raise ex.TypeError("`velocity` should be a numpy array")

        self._velocity = velocity

    @property
    def loudness(self) -> np.ndarray:
        """Array of loudnesses."""

        return self._loudness

    @loudness.setter
    def loudness(self, loudness: np.ndarray) -> None:
        if not isinstance(loudness, np.ndarray):
            raise ex.TypeError("`loudness` should be a numpy array")

        self._loudness = loudness

    @property
    def pulse_rate(self) -> np.ndarray:
        """Array of pulse rates."""

        return self._pulse_rate

    @pulse_rate.setter
    def pulse_rate(self, pulse_rate: np.ndarray) -> None:
        if not isinstance(pulse_rate, np.ndarray):
            raise ex.TypeError("`pulse_rate` should be a numpy array")

        self._pulse_rate = pulse_rate

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Arrays of frequencies, velocities, loudnesses and pulse rates
        self.frequency = rnd.generate_uniform_random_number(
            self.f_min, self.f_max, space.n_agents
        )
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )
        self.loudness = rnd.generate_uniform_random_number(0, self.A, space.n_agents)
        self.pulse_rate = rnd.generate_uniform_random_number(0, self.r, space.n_agents)

    def update(self, space: Space, function: Function, iteration: int) -> None:
        """Wraps Bat Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.

        """

        # Declares alpha constant
        alpha = 0.9

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Updates frequency (eq. 2)
            # Note that we have to apply (min - max) instead of (max - min) or it will not converge
            beta = rnd.generate_uniform_random_number()
            self.frequency[i] = self.f_min + (self.f_min - self.f_max) * beta

            # Updates velocity (eq. 3)
            self.velocity[i] += (
                agent.position - space.best_agent.position
            ) * self.frequency[i]

            # Updates agent's position (eq. 4)
            agent.position += self.velocity[i]

            # Generates random uniform and gaussian numbers
            p = rnd.generate_uniform_random_number()
            e = rnd.generate_gaussian_random_number()

            # Checks if probability is bigger than current pulse rate
            if p > self.pulse_rate[i]:
                # Performs a local random walk (eq. 5)
                # We apply 0.001 to limit the step size
                agent.position = space.best_agent.position + 0.001 * e * np.mean(
                    self.loudness
                )

            # Checks agent limits
            agent.clip_by_bound()

            # Evaluates agent
            agent.fit = function(agent.position)

            # Checks if probability is smaller than loudness and if fit is better
            if p < self.loudness[i] and agent.fit < space.best_agent.fit:
                # Copies the new solution to space's best agent
                space.best_agent = copy.deepcopy(agent)

                # Increasing pulse rate (eq. 6 - left)
                self.pulse_rate[i] = self.r * (1 - np.exp(-alpha * iteration))

                # Decreasing loudness (eq. 6 - right)
                self.loudness[i] = self.A * alpha

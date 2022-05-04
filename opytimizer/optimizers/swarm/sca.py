"""Sine Cosine Algorithm.
"""

from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class SCA(Optimizer):
    """A SCA class, inherited from Optimizer.

    This is the designed class to define SCA-related
    variables and methods.

    References:
        S. Mirjalili. SCA: A Sine Cosine Algorithm for solving optimization problems.
        Knowledge-Based Systems (2016).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> SCA.")

        # Overrides its parent class with the receiving params
        super(SCA, self).__init__()

        # Minimum function range
        self.r_min = 0

        # Maximum function range
        self.r_max = 2

        # Constant for defining the next position's region
        self.a = 3

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def r_min(self) -> float:
        """Minimum function range."""

        return self._r_min

    @r_min.setter
    def r_min(self, r_min: float) -> None:
        if not isinstance(r_min, (float, int)):
            raise e.TypeError("`r_min` should be a float or integer")
        if r_min < 0:
            raise e.ValueError("`r_min` should be >= 0")

        self._r_min = r_min

    @property
    def r_max(self) -> float:
        """Maximum function range."""

        return self._r_max

    @r_max.setter
    def r_max(self, r_max: float) -> None:
        if not isinstance(r_max, (float, int)):
            raise e.TypeError("`r_max` should be a float or integer")
        if r_max < 0:
            raise e.ValueError("`r_max` should be >= 0")
        if r_max < self.r_min:
            raise e.ValueError("`r_max` should be >= `r_min`")

        self._r_max = r_max

    @property
    def a(self) -> float:
        """Loudness parameter."""

        return self._a

    @a.setter
    def a(self, a: float) -> None:
        if not isinstance(a, (float, int)):
            raise e.TypeError("`a` should be a float or integer")
        if a < 0:
            raise e.ValueError("`a` should be >= 0")

        self._a = a

    def _update_position(
        self,
        agent_position: np.ndarray,
        best_position: np.ndarray,
        r1: float,
        r2: float,
        r3: float,
        r4: float,
    ) -> np.ndarray:
        """Updates a single particle position over a single variable (eq. 3.3).

        Args:
            agent_position: Agent's current position.
            best_position: Global best position.
            r1: Controls the next position's region.
            r2: Defines how far the movement should be.
            r3: Random weight for emphasizing or deemphasizing the movement.
            r4: Random number to decide whether sine or cosine should be used.

        Returns:
            (np.ndarray): A new position.

        """

        # If random number is smaller than threshold
        if r4 < 0.5:
            # Updates the position using sine
            new_position = agent_position + r1 * np.sin(r2) * np.fabs(
                r3 * best_position - agent_position
            )

        # If the random number is bigger than threshold
        else:
            # Updates the posistion using cosine
            new_position = agent_position + r1 * np.cos(r2) * np.fabs(
                r3 * best_position - agent_position
            )

        return new_position

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        """Wraps Sine Cosine Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Adaptively changing the r1 parameter, which controls the next position's region
        r1 = self.a - (iteration * self.a / n_iterations)

        # The r2 parameter defines how far the movement should be
        r2 = r.generate_uniform_random_number(0, 2 * np.pi)

        # A random weight for emphasizing or deemphasizing the movement
        r3 = r.generate_uniform_random_number(self.r_min, self.r_max)

        # A random number to decide whether sine or cosine should be used
        r4 = r.generate_uniform_random_number()

        # Iterates through all agents
        for agent in space.agents:
            # Updates agent's position
            agent.position = self._update_position(
                agent.position, space.best_agent.position, r1, r2, r3, r4
            )

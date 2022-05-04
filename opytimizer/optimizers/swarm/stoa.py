"""Sooty Tern Optimization Algorithm.
"""

from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class STOA(Optimizer):
    """An STOA class, inherited from Optimizer.

    This is the designed class to define STOA-related
    variables and methods.

    References:
        G. Dhiman and A. Kaur. STOA: A bio-inspired based optimization algorithm for industrial engineering problems.
        Engineering Applications of Artificial Intelligence (2019).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> STOA.")

        # Overrides its parent class with the receiving params
        super(STOA, self).__init__()

        # Controlling variable
        self.Cf = 2.0

        # Spiral shape first constant
        self.u = 1.0

        # Spiral shape second constant
        self.v = 1.0

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def Cf(self) -> float:
        """Controlling variable."""

        return self._Cf

    @Cf.setter
    def Cf(self, Cf: float) -> None:
        if not isinstance(Cf, (float, int)):
            raise e.TypeError("`Cf` should be a float or integer")
        if Cf < 0:
            raise e.ValueError("`Cf` should be >= 0")

        self._Cf = Cf

    @property
    def u(self) -> float:
        """Spiral shape first constant."""

        return self._u

    @u.setter
    def u(self, u: float) -> None:
        if not isinstance(u, (float, int)):
            raise e.TypeError("`u` should be a float or integer")
        if u < 0:
            raise e.ValueError("`u` should be >= 0")

        self._u = u

    @property
    def v(self) -> float:
        """Spiral shape second constant."""

        return self._v

    @v.setter
    def v(self, v: float) -> None:
        if not isinstance(v, (float, int)):
            raise e.TypeError("`v` should be a float or integer")
        if v < 0:
            raise e.ValueError("`v` should be >= 0")

        self._v = v

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        """Wraps Sooty Tern Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Calculates the movement of search space (eq. 2)
        Sa = self.Cf - (iteration * (self.Cf / n_iterations))

        # Calculates the exploration variable (eq. 4)
        Cb = 0.5 * r.generate_uniform_random_number()

        # Iterates through all agents
        for agent in space.agents:
            # Calculates the collision avoidance (eq. 1)
            C = Sa * agent.position

            # Calculates the convergence towards the best agent (eq. 3)
            M = Cb * (space.best_agent.position - agent.position)

            # Calculates the gap between agent and best agent (eq. 5)
            D = C + M

            # Defines the spiral radius (eq. 9)
            k = r.generate_uniform_random_number(0, 2 * np.pi)
            R = self.u * np.exp(k * self.v)

            # Calculates the spiral movement (eq. 6, 7 and 8)
            i = r.generate_uniform_random_number(0, k)
            x = R * np.sin(i)
            y = R * np.cos(i)
            z = R * i

            # Updates the agent's position (eq. 10)
            agent.position = (D * (x + y + z)) * space.best_agent.position

"""Wind Driven Optimization.
"""

from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class WDO(Optimizer):
    """A WDO class, inherited from Optimizer.

    This is the designed class to define WDO-related
    variables and methods.

    References:
        Z. Bayraktar et al. The wind driven optimization technique and its application in electromagnetics.
        IEEE transactions on antennas and propagation (2013).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> WDO.")

        # Overrides its parent class with the receiving params
        super(WDO, self).__init__()

        # Maximum velocity
        self.v_max = 0.3

        # Friction coefficient
        self.alpha = 0.8

        # Gravitational force coefficient
        self.g = 0.6

        # Coriolis force
        self.c = 1.0

        # Pressure constant
        self.RT = 1.5

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def v_max(self) -> float:
        """Maximum velocity."""

        return self._v_max

    @v_max.setter
    def v_max(self, v_max: float) -> None:
        if not isinstance(v_max, (float, int)):
            raise e.TypeError("`v_max` should be a float or integer")
        if v_max < 0:
            raise e.ValueError("`v_max` should be >= 0")

        self._v_max = v_max

    @property
    def alpha(self) -> float:
        """Friction coefficient."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        if alpha < 0 or alpha > 1:
            raise e.ValueError("`alpha` should be between 0 and 1")

        self._alpha = alpha

    @property
    def g(self) -> float:
        """Gravitational force coefficient."""

        return self._g

    @g.setter
    def g(self, g: float) -> None:
        if not isinstance(g, (float, int)):
            raise e.TypeError("`g` should be a float or integer")
        if g < 0:
            raise e.ValueError("`g` should be >= 0")

        self._g = g

    @property
    def c(self) -> float:
        """Coriolis force."""

        return self._c

    @c.setter
    def c(self, c: float) -> None:
        if not isinstance(c, (float, int)):
            raise e.TypeError("`c` should be a float or integer")
        if c < 0:
            raise e.ValueError("`c` should be >= 0")

        self._c = c

    @property
    def RT(self) -> float:
        """Pressure constant."""

        return self._RT

    @RT.setter
    def RT(self, RT: float) -> None:
        if not isinstance(RT, (float, int)):
            raise e.TypeError("`RT` should be a float or integer")
        if RT < 0:
            raise e.ValueError("`RT` should be >= 0")

        self._RT = RT

    @property
    def velocity(self) -> np.ndarray:
        """Array of velocities."""

        return self._velocity

    @velocity.setter
    def velocity(self, velocity: np.ndarray) -> None:
        if not isinstance(velocity, np.ndarray):
            raise e.TypeError("`velocity` should be a numpy array")

        self._velocity = velocity

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Arrays of velocities
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def update(self, space: Space, function: Function) -> None:
        """Wraps Wind Driven Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A function object.

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Generates a random index based on the number of agents
            index = r.generate_integer_random_number(0, len(space.agents))

            # Updates velocity (eq. 15)
            self.velocity[i] = (
                (1 - self.alpha) * self.velocity[i]
                - self.g * agent.position
                + (
                    self.RT
                    * np.abs(1 / (index + 1) - 1)
                    * (space.best_agent.position - agent.position)
                )
                + (self.c * self.velocity[index] / (index + 1))
            )

            # Clips the velocity values between [-v_max, v_max]
            self.velocity = np.clip(self.velocity, -self.v_max, self.v_max)

            # Updates agent's position (eq. 16)
            agent.position += self.velocity[i]

            # Checks agent limits
            agent.clip_by_bound()

            # Evaluates agent
            agent.fit = function(agent.position)

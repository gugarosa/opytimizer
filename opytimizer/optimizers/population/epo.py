"""Emperor Penguin Optimizer.
"""

from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class EPO(Optimizer):
    """An EPO class, inherited from Optimizer.

    This is the designed class to define EPO-related
    variables and methods.

    References:
        G. Dhiman and V. Kumar. Emperor penguin optimizer: A bio-inspired algorithm for engineering problems.
        Knowledge-Based Systems (2018).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> EPO.")

        # Overrides its parent class with the receiving params
        super(EPO, self).__init__()

        # Exploration control parameter
        self.f = 2.0

        # Exploitation control parameter
        self.l = 1.5

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def f(self) -> float:
        """Exploration control parameter."""

        return self._f

    @f.setter
    def f(self, f: float) -> None:
        if not isinstance(f, (float, int)):
            raise e.TypeError("`f` should be a float or integer")

        self._f = f

    @property
    def l(self) -> float:
        """Exploitation control parameter."""

        return self._l

    @l.setter
    def l(self, l: float) -> None:
        if not isinstance(l, (float, int)):
            raise e.TypeError("`l` should be a float or integer")

        self._l = l

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        """Wraps Emperor Penguin Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Iterates through every agent
        for agent in space.agents:
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

            # Calculates the temperature profile (eq. 7)
            T_p = T - n_iterations / (iteration - n_iterations)

            # Calculates the polygon grid accuracy (eq. 10)
            P_grid = np.fabs(space.best_agent.position - agent.position)

            # Generates a uniform random number and the `C` coefficient
            r1 = r.generate_uniform_random_number()
            C = r.generate_uniform_random_number(size=agent.n_variables)

            # Calculates the avoidance coefficient (eq. 9)
            A = 2 * (T_p + P_grid) * r1 - T_p

            # Calculates the social forces of emperor penguin (eq. 12)
            S = (
                np.fabs(self.f * np.exp(-iteration / self.l) - np.exp(-iteration))
            ) ** 2

            # Calculates the distance between current agent and emperor penguin (eq. 8)
            D_ep = np.fabs(S * space.best_agent.position - C * agent.position)

            # Updates current agent's position (eq. 13)
            agent.position = space.best_agent.position - A * D_ep

"""Arithmetic Optimization Algorithm.
"""

from typing import Any, Dict, Optional

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class AOA(Optimizer):
    """An AOA class, inherited from Optimizer.

    This is the designed class to define AOA-related
    variables and methods.

    References:
        L. Abualigah et al. The Arithmetic Optimization Algorithm.
        Computer Methods in Applied Mechanics and Engineering (2021).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> AOA.")

        # Overrides its parent class with the receiving params
        super(AOA, self).__init__()

        # Minimum accelerated function
        self.a_min = 0.2

        # Maximum accelerated function
        self.a_max = 1.0

        # Sensitive parameter
        self.alpha = 5.0

        # Control parameter
        self.mu = 0.499

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def a_min(self) -> float:
        """Minimum accelerated function."""

        return self._a_min

    @a_min.setter
    def a_min(self, a_min: float) -> None:
        if not isinstance(a_min, (float, int)):
            raise e.TypeError("`a_min` should be a float or integer")
        if a_min < 0:
            raise e.ValueError("`a_min` should be >= 0")

        self._a_min = a_min

    @property
    def a_max(self) -> float:
        """Maximum accelerated function."""

        return self._a_max

    @a_max.setter
    def a_max(self, a_max: float) -> None:
        if not isinstance(a_max, (float, int)):
            raise e.TypeError("`a_max` should be a float or integer")
        if a_max < 0:
            raise e.ValueError("`a_max` should be >= 0")
        if a_max < self.a_min:
            raise e.ValueError("`a_max` should be >= `a_min`")

        self._a_max = a_max

    @property
    def alpha(self) -> float:
        """Sensitive parameter."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        if alpha < 0:
            raise e.ValueError("`alpha` should be >= 0")

        self._alpha = alpha

    @property
    def mu(self) -> float:
        """Control parameter."""

        return self._mu

    @mu.setter
    def mu(self, mu: float) -> None:
        if not isinstance(mu, (float, int)):
            raise e.TypeError("`mu` should be a float or integer")
        if mu < 0:
            raise e.ValueError("`mu` should be >= 0")

        self._mu = mu

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        """Wraps Arithmetic Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Calculates math optimizer accelarated coefficient (eq. 2)
        MOA = self.a_min + iteration * ((self.a_max - self.a_min) / n_iterations)

        # Calculates math optimizer probability (eq. 4)
        MOP = 1 - (iteration ** (1 / self.alpha) / n_iterations ** (1 / self.alpha))

        # Iterates through all agents
        for agent in space.agents:
            # Iterates through all variables
            for j in range(agent.n_variables):
                # Generates random probability
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
                        agent.position[j] = (
                            space.best_agent.position[j]
                            / (MOP + c.EPSILON)
                            * search_partition
                        )

                    # If probability is smaller than 0.5
                    else:
                        # Updates position with (eq. 3 - bottom)
                        agent.position[j] = (
                            space.best_agent.position[j] * MOP * search_partition
                        )

                # If probability is smaller than MOA
                else:
                    # Generates an extra probability
                    r3 = r.generate_uniform_random_number()

                    # If probability is bigger than 0.5
                    if r3 > 0.5:
                        # Updates position with (eq. 5 - top)
                        agent.position[j] = (
                            space.best_agent.position[j] - MOP * search_partition
                        )

                    # If probability is smaller than 0.5
                    else:
                        # Updates position with (eq. 5 - bottom)
                        agent.position[j] = (
                            space.best_agent.position[j] + MOP * search_partition
                        )

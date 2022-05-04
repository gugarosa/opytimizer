"""Hill-Climbing.
"""

from typing import Any, Dict, Optional

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class HC(Optimizer):
    """An HC class, inherited from Optimizer.

    This is the designed class to define HC-related
    variables and methods.

    References:
        S. Skiena. The Algorithm Design Manual (2010).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> HC.")

        # Overrides its parent class with the receiving params
        super(HC, self).__init__()

        # Mean of noise distribution
        self.r_mean = 0

        # Variance of noise distribution
        self.r_var = 0.1

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def r_mean(self) -> float:
        """Mean of noise distribution."""

        return self._r_mean

    @r_mean.setter
    def r_mean(self, r_mean: float) -> None:
        if not isinstance(r_mean, (float, int)):
            raise e.TypeError("`r_mean` should be a float or integer")

        self._r_mean = r_mean

    @property
    def r_var(self) -> float:
        """Variance of noise distribution."""

        return self._r_var

    @r_var.setter
    def r_var(self, r_var: float) -> None:
        if not isinstance(r_var, (float, int)):
            raise e.TypeError("`r_var` should be a float or integer")
        if r_var < 0:
            raise e.ValueError("`r_var` should be >= 0")

        self._r_var = r_var

    def update(self, space: Space) -> None:
        """Wraps Hill Climbing over all agents and variables (p. 252).

        Args:
            space: Space containing agents and update-related information.

        """

        # Iterates through all agents
        for agent in space.agents:
            # Creates a gaussian noise vector
            noise = r.generate_gaussian_random_number(
                self.r_mean, self.r_var, size=(agent.n_variables, agent.n_dimensions)
            )

            # Updates agent's position
            agent.position += noise

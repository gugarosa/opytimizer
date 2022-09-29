"""Firefly Algorithm.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class FA(Optimizer):
    """A FA class, inherited from Optimizer.

    This is the designed class to define FA-related
    variables and methods.

    References:
        X.-S. Yang. Firefly algorithms for multimodal optimization.
        International symposium on stochastic algorithms (2009).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> FA.")

        super(FA, self).__init__()

        self.alpha = 0.5
        self.beta = 0.2
        self.gamma = 1.0

        self.build(params)

        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        """Randomization parameter."""

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
        """Attractiveness parameter."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        if beta < 0:
            raise e.ValueError("`beta` should be >= 0")

        self._beta = beta

    @property
    def gamma(self) -> float:
        """Light absorption coefficient."""

        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float) -> None:
        if not isinstance(gamma, (float, int)):
            raise e.TypeError("`gamma` should be a float or integer")
        if gamma < 0:
            raise e.ValueError("`gamma` should be >= 0")

        self._gamma = gamma

    def update(self, space: Space, n_iterations: int) -> None:
        """Wraps Firefly Algorithm over all agents and variables (eq. 3-9).

        Args:
            space: Space containing agents and update-related information.
            n_iterations: Maximum number of iterations.

        """

        delta = 1 - ((10e-4) / 0.9) ** (1 / n_iterations)
        self.alpha *= 1 - delta

        temp_agents = copy.deepcopy(space.agents)

        for agent in space.agents:
            for temp in temp_agents:
                # Distance is calculated by an euclidean distance between 'i' and 'j' (eq. 8)
                distance = g.euclidean_distance(agent.position, temp.position)

                if agent.fit > temp.fit:
                    # Recalculate the attractiveness (eq. 6)
                    beta = self.beta * np.exp(-self.gamma * distance)

                    # Updates agent's position (eq. 9)
                    r1 = r.generate_uniform_random_number()
                    agent.position = beta * (
                        temp.position + agent.position
                    ) + self.alpha * (r1 - 0.5)

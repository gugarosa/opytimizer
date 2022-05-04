"""Algorithm of the Innovative Gunner.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class AIG(Optimizer):
    """An AIG class, inherited from Optimizer.

    This is the designed class to define AIG-related
    variables and methods.

    References:
        P. Pijarski and P. Kacejko.
        A new metaheuristic optimization method: the algorithm of the innovative gunner (AIG).
        Engineering Optimization (2019).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> AIG.")

        # Overrides its parent class with the receiving params
        super(AIG, self).__init__()

        # First maximum correction angle
        self.alpha = np.pi

        # Second maximum correction angle
        self.beta = np.pi

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        """First maximum correction angle."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        if alpha < 0 or alpha > np.pi * 2:
            raise e.ValueError("`alpha` should be between 0 and 2PI")

        self._alpha = alpha

    @property
    def beta(self) -> float:
        """Second maximum correction angle."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        if beta < 0 or beta > np.pi * 2:
            raise e.ValueError("`beta` should be between 0 and 2PI")

        self._beta = beta

    def update(self, space: Space, function: Function) -> None:
        """Wraps Algorithm of the Innovative Gunner over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Calculates the maximum correction angles (eq. 18)
        a = r.generate_uniform_random_number()
        alpha_max = self.alpha * a
        beta_max = self.beta * a

        # Iterates through all agents
        for agent in space.agents:
            # Makes a deep copy of current agent
            a = copy.deepcopy(agent)

            # Samples correction angles
            alpha = r.generate_gaussian_random_number(
                0, alpha_max / 3, (agent.n_variables, agent.n_dimensions)
            )
            beta = r.generate_gaussian_random_number(
                0, beta_max / 3, (agent.n_variables, agent.n_dimensions)
            )

            # Calculates correction functions (eq. 16 and 17)
            g_alpha = np.where(alpha < 0, np.cos(alpha), 1 / np.cos(alpha))
            g_beta = np.where(beta < 0, np.cos(beta), 1 / np.cos(beta))

            # Updates temporary agent's position (eq. 15)
            a.position *= g_alpha * g_beta

            # Checks agent's limits
            a.clip_by_bound()

            # Re-evaluates the temporary agent
            a.fit = function(a.position)

            # If temporary agent's fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Replace its position and fitness
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

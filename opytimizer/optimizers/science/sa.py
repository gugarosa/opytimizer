"""Simulated Annealing.
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


class SA(Optimizer):
    """A SA class, inherited from Optimizer.

    This is the designed class to define SA-related
    variables and methods.

    References:
        A. Khachaturyan, S. Semenovsovskaya and B. Vainshtein.
        The thermodynamic approach to the structure analysis of crystals.
        Acta Crystallographica (1981).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> SA.")

        # Overrides its parent class with the receiving params
        super(SA, self).__init__()

        # System's temperature
        self.T = 100

        # Temperature decay
        self.beta = 0.999

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def T(self) -> float:
        """System's temperature."""

        return self._T

    @T.setter
    def T(self, T: float) -> None:
        if not isinstance(T, (float, int)):
            raise e.TypeError("`T` should be a float or integer")
        if T < 0:
            raise e.ValueError("`T` should be >= 0")

        self._T = T

    @property
    def beta(self) -> float:
        """Temperature decay."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        if beta < 0:
            raise e.ValueError("`beta` should be >= 0")

        self._beta = beta

    def update(self, space: Space, function: Function) -> None:
        """Wraps Simulated Annealing over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A function object.

        """

        # Iterates through all agents
        for agent in space.agents:
            # Mimics its position
            a = copy.deepcopy(agent)

            # Generates a random noise from a gaussian distribution
            noise = r.generate_gaussian_random_number(
                0, 0.1, size=((agent.n_variables, agent.n_dimensions))
            )

            # Applies the noise
            a.position += noise

            # Checks agent's limits
            a.clip_by_bound()

            # Calculates the fitness for the temporary position
            a.fit = function(a.position)

            # Generates an uniform random number
            r1 = r.generate_uniform_random_number()

            # If new fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Copies its position and fitness to the agent
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

            # Checks if state should be updated or not
            elif r1 < np.exp(-(a.fit - agent.fit) / self.T):
                # Copies its position and fitness to the agent
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

        # Decay the temperature
        self.T *= self.beta

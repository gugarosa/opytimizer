"""Darcy Optimization Algorithm.
"""

from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as rnd
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class DOA(Optimizer):
    """A DOA class, inherited from Optimizer.

    This is the designed class to define DOA-related
    variables and methods.

    References:
        F. Demir et al. A survival classification method for hepatocellular carcinoma patients
        with chaotic Darcy optimization method based feature selection.
        Medical Hypotheses (2020).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> DOA.")

        # Overrides its parent class with the receiving params
        super(DOA, self).__init__()

        # Chaos multiplier
        self.r = 1.0

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def r(self) -> float:
        """Chaos multiplier."""

        return self._r

    @r.setter
    def r(self, r: float) -> None:
        if not isinstance(r, (float, int)):
            raise e.TypeError("`r` should be a float or integer")
        if r < 0:
            raise e.ValueError("`r` should be >= 0")

        self._r = r

    @property
    def chaotic_map(self) -> np.ndarray:
        """Array of chaotic maps."""

        return self._chaotic_map

    @chaotic_map.setter
    def chaotic_map(self, chaotic_map: np.ndarray) -> None:
        if not isinstance(chaotic_map, np.ndarray):
            raise e.TypeError("`chaotic_map` should be a numpy array")

        self._chaotic_map = chaotic_map

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Array of chaotic maps
        self.chaotic_map = np.zeros((space.n_agents, space.n_variables))

    def _calculate_chaotic_map(self, lb: float, ub: float) -> float:
        """Calculates the chaotic map (eq. 3).

        Args:
            lb: Lower bound value.
            ub: Upper bound value.

        Returns:
            (float): A new value for the chaotic map.

        """

        # Generates a uniform random number between variable's bounds
        r1 = rnd.generate_uniform_random_number(lb, ub)

        # Calculates the chaotic map (eq. 3)
        c_map = self.r * r1 * (1 - r1) + ((4 - self.r) * np.sin(np.pi * r1)) / 4

        return c_map

    def update(self, space: Space) -> None:
        """Wraps Darcy Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Iterates through all decision variables
            for j, (lb, ub) in enumerate(zip(agent.lb, agent.ub)):
                # Generates a chaotic map
                c_map = self._calculate_chaotic_map(lb, ub)

                # Updates the agent's position (eq. 6)
                agent.position[j] += (
                    (
                        2
                        * (space.best_agent.position[j] - agent.position[j])
                        / (c_map - self.chaotic_map[i][j])
                    )
                    * (ub - lb)
                    / len(space.agents)
                )

                # Updates current chaotic map with newer value
                self.chaotic_map[i][j] = c_map

                # Checks if position has exceed the bounds
                if (agent.position[j] < lb) or (agent.position[j] > ub):
                    # If yes, replace its value with the proposed equation (eq. 7)
                    agent.position[j] = space.best_agent.position[j] * c_map

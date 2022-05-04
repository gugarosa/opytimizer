"""Grasshopper Optimization Algorithm.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.general as g
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class GOA(Optimizer):
    """A GOA class, inherited from Optimizer.

    This is the designed class to define GOA-related
    variables and methods.

    References:
        S. Saremi, S. Mirjalili and A. Lewis. Grasshopper Optimisation Algorithm: Theory and application.
        Advances in Engineering Software (2017).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> GOA.")

        # Overrides its parent class with the receiving params
        super(GOA, self).__init__()

        # Minimum comfort zone
        self.c_min = 0.00001

        # Maximum comfort zone
        self.c_max = 1

        # Intensity of attraction
        self.f = 0.5

        # Attractive length scale
        self.l = 1.5

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def c_min(self) -> float:
        """Minimum comfort zone."""

        return self._c_min

    @c_min.setter
    def c_min(self, c_min: float) -> None:
        if not isinstance(c_min, (float, int)):
            raise e.TypeError("`c_min` should be a float or integer")
        if c_min < 0:
            raise e.ValueError("`c_min` should be >= 0")

        self._c_min = c_min

    @property
    def c_max(self) -> float:
        """Maximum comfort zone."""

        return self._c_max

    @c_max.setter
    def c_max(self, c_max: float) -> None:
        if not isinstance(c_max, (float, int)):
            raise e.TypeError("`c_max` should be a float or integer")
        if c_max < self.c_min:
            raise e.ValueError("`c_max` should be >= `c_min`")

        self._c_max = c_max

    @property
    def f(self) -> float:
        """Intensity of attraction."""

        return self._f

    @f.setter
    def f(self, f: float) -> None:
        if not isinstance(f, (float, int)):
            raise e.TypeError("`f` should be a float or integer")
        if f < 0:
            raise e.ValueError("`f` should be >= 0")

        self._f = f

    @property
    def l(self) -> float:
        """Attractive length scale."""

        return self._l

    @l.setter
    def l(self, l: float) -> None:
        if not isinstance(l, (float, int)):
            raise e.TypeError("`l` should be a float or integer")
        if l < 0:
            raise e.ValueError("`l` should be >= 0")

        self._l = l

    def _social_force(self, r: np.ndarray) -> np.ndarray:
        """Calculates the social force based on an input value.

        Args:
            r: Array of values.

        Returns:
            (np.ndarray): The social force based on the input value.

        """

        # Calculates the social force (eq. 2.3)
        s = self.f * np.exp(-r / self.l) - np.exp(-r)

        return s

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Grasshopper Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Calculates the comfort coefficient (eq. 2.8)
        comfort = self.c_max - iteration * ((self.c_max - self.c_min) / n_iterations)

        # Copies a temporary list for iterating purposes
        temp_agents = copy.deepcopy(space.agents)

        # Iterates through 'i' agents
        for agent in space.agents:
            # Initializes the total comfort as zero
            total_comfort = np.zeros((agent.n_variables, agent.n_dimensions))

            # Iterates through 'j' agents
            for temp in temp_agents:
                # Distance is calculated by an euclidean distance between 'i' and 'j'
                distance = g.euclidean_distance(agent.position, temp.position)

                # Calculates the unitary vector
                unit = (temp.position - agent.position) / (distance + c.EPSILON)

                # Calculates the social force between agents
                s = self._social_force(2 + np.fmod(distance, 2))

                # Expands the upper and lower bounds
                ub = np.expand_dims(agent.ub, -1)
                lb = np.expand_dims(agent.lb, -1)

                # Sums the current comfort to the total one
                total_comfort += comfort * ((ub - lb) / 2) * s * unit

            # Updates the agent's position (eq. 2.7)
            agent.position = comfort * total_comfort + space.best_agent.position

            # Checks the agent's limits
            agent.clip_by_bound()

            # Evaluates the new agent's position
            agent.fit = function(agent.position)

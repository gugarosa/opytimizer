"""Fruit-Fly Optimization Algorithm.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class FFOA(Optimizer):
    """A FFOA class, inherited from Optimizer.

    This is the designed class to define FFOA-related
    variables and methods.

    References:
        W.-T. Pan. A new Fruit Fly Optimization Algorithm: Taking the financial distress model as an example.
        Knowledge-Based Systems (2012).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> FFOA.")

        # Overrides its parent class with the receiving params
        super(FFOA, self).__init__()

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def x_axis(self) -> List[Agent]:
        """`x` axis."""

        return self._x_axis

    @x_axis.setter
    def x_axis(self, x_axis: List[Agent]) -> None:
        if not isinstance(x_axis, list):
            raise e.TypeError("`x_axis` should be a list")

        self._x_axis = x_axis

    @property
    def y_axis(self) -> List[Agent]:
        """`y` axis."""

        return self._y_axis

    @y_axis.setter
    def y_axis(self, y_axis: List[Agent]) -> None:
        if not isinstance(y_axis, list):
            raise e.TypeError("`y_axis` should be a list")

        self._y_axis = y_axis

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Lists of `x` and `y` axis (eq. 1)
        self.x_axis = copy.deepcopy(space.agents)
        self.y_axis = copy.deepcopy(space.agents)

    def update(self, space: Space, function: Function) -> None:
        """Wraps Fruit-Fly Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Iterates through all agents and their axis
        for a, x_axis, y_axis in zip(space.agents, self.x_axis, self.y_axis):
            # Generates random numbers
            r1 = r.generate_uniform_random_number()
            r2 = r.generate_uniform_random_number()

            # Shakes the `x` and `y` axis positions (eq. 2)
            x = x_axis.position + r1
            y = y_axis.position + r2

            # Calculates the distance between axis (eq. 3 - top)
            distance = np.sqrt(x**2 + y**2)

            # Calculates the smell's position (eq. 3 - bottom)
            s = 1 / (distance + c.EPSILON)

            # Evaluates the smell's position (eq. 4)
            smell = function(s)

            # If smell's fitness is better than agent's fitness
            if smell < a.fit:
                # Updates its corresponding `axis` positions (eq. 6)
                x_axis.position = copy.deepcopy(x)
                y_axis.position = copy.deepcopy(y)

                # Updates the agent's position and fitness
                a.position = copy.deepcopy(s)
                a.fit = copy.deepcopy(smell)

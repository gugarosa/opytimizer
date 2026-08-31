"""Fruit-Fly Optimization Algorithm."""

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


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

        super(FFOA, self).__init__()

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Lists of `x` and `y` axis (eq. 1)
        self.x_axis = copy.deepcopy(space.agents)
        self.y_axis = copy.deepcopy(space.agents)

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Fruit-Fly Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        for a, x_axis, y_axis in zip(space.agents, self.x_axis, self.y_axis):
            r1 = np.random.uniform(0.0, 1.0, 1)
            r2 = np.random.uniform(0.0, 1.0, 1)

            # Shakes the `x` and `y` axis positions (eq. 2)
            x = x_axis.position + r1
            y = y_axis.position + r2

            # Calculates the distance between axis (eq. 3 - top)
            distance = np.sqrt(x**2 + y**2)

            # Calculates the smell's position (eq. 3 - bottom)
            s = 1 / (distance + c.EPSILON)

            # Evaluates the smell's position (eq. 4)
            smell = function(s)

            if smell < a.fit:
                # Updates its corresponding `axis` positions (eq. 6)
                x_axis.position = copy.deepcopy(x)
                y_axis.position = copy.deepcopy(y)

                a.position = copy.deepcopy(s)
                a.fit = copy.deepcopy(smell)

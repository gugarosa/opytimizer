# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Fruit-Fly Optimization Algorithm.

References:
    W.-T. Pan. A new Fruit Fly Optimization Algorithm: Taking the financial distress model as an example.
    Knowledge-Based Systems (2012).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class FFOA(Optimizer):
    """Search by evaluating fruit-fly smell positions from two coordinate populations.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize fruit-fly search configuration.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            No algorithm-specific parameter defaults are defined.
            Compilation deep-copies agents into ``x_axis`` and ``y_axis``, which retain improving coordinates.

        """

        super(FFOA, self).__init__()

        self.build(params)

    def compile(self, space: Space) -> None:
        # Lists of `x` and `y` axis (eq. 1)
        self.x_axis = copy.deepcopy(space.agents)
        self.y_axis = copy.deepcopy(space.agents)

    def update(self, space: Space, function: Callable) -> None:
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

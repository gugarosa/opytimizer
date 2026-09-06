# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Firefly Algorithm.

Movement and brightness-based attraction follow equations 3-9 of the reference.

References:
    X.-S. Yang. Firefly algorithms for multimodal optimization.
    International symposium on stochastic algorithms (2009).

"""

import copy
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class FA(Optimizer):
    """Move fireflies toward brighter neighbors with decaying random perturbations.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure firefly attraction and randomization.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``alpha`` (0.5) scales random displacement and is reduced in place on each update.
            ``beta`` (0.2) is base attractiveness, and ``gamma`` (1.0) controls exponential distance attenuation.

        """

        super(FA, self).__init__()

        self.alpha = 0.5
        self.beta = 0.2
        self.gamma = 1.0

        self.build(params)

    def update(self, space: Space, n_iterations: int) -> None:
        delta = 1 - ((10e-4) / 0.9) ** (1 / n_iterations)
        self.alpha *= 1 - delta

        temp_agents = copy.deepcopy(space.agents)

        for agent in space.agents:
            for temp in temp_agents:
                # Distance is calculated by an euclidean distance between 'i' and 'j' (eq. 8)
                distance = np.linalg.norm(agent.position - temp.position)

                if agent.fit > temp.fit:
                    # Recalculate the attractiveness (eq. 6)
                    beta = self.beta * np.exp(-self.gamma * distance)

                    # Updates agent's position (eq. 9)
                    r1 = np.random.uniform(0.0, 1.0, 1)
                    agent.position = beta * (temp.position + agent.position) + self.alpha * (r1 - 0.5)

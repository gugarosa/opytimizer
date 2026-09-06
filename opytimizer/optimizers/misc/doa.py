# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Darcy Optimization Algorithm.

Compilation stores the previous chaotic-map value for each agent and variable.
Updates use changes in the map to move toward the best position and replace out-of-bound values.

References:
    F. Demir et al. A survival classification method for hepatocellular carcinoma patients
    with chaotic Darcy optimization method based feature selection. Medical Hypotheses (2020).

"""

from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class DOA(Optimizer):
    """Optimize a population using chaotic Darcy updates.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the chaotic-map control coefficient.

        Args:
            params: Overrides for the supported optimizer parameters.

        Notes:
            The supported key is ``r`` (logistic versus sinusoidal map coefficient, 1.0).

        """

        super(DOA, self).__init__()

        self.r = 1.0

        self.build(params)

    def compile(self, space: Space) -> None:
        self.chaotic_map = np.zeros((space.n_agents, space.n_variables))

    def _calculate_chaotic_map(self, lb: float, ub: float) -> float:
        r1 = np.random.uniform(lb, ub)

        # Equation 3
        c_map = self.r * r1 * (1 - r1) + ((4 - self.r) * np.sin(np.pi * r1)) / 4

        return c_map

    def update(self, space: Space) -> None:
        for i, agent in enumerate(space.agents):
            for j, (lb, ub) in enumerate(zip(agent.lb, agent.ub)):
                c_map = self._calculate_chaotic_map(lb, ub)

                # Equation 6
                agent.position[j] += (
                    (2 * (space.best_agent.position[j] - agent.position[j]) / (c_map - self.chaotic_map[i][j]))
                    * (ub - lb)
                    / len(space.agents)
                )

                self.chaotic_map[i][j] = c_map

                if (agent.position[j] < lb) or (agent.position[j] > ub):
                    # Equation 7
                    agent.position[j] = space.best_agent.position[j] * c_map

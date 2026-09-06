# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Butterfly Optimization Algorithm.

References:
    S. Arora and S. Singh. Butterfly optimization algorithm: a novel approach for global optimization.
    Soft Computing (2019).

"""

from typing import Any

import numpy as np

import opytimizer.math.random as r
from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class BOA(Optimizer):
    """Move butterflies using fragrance-driven global and local attraction.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure butterfly fragrance and movement selection.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``c`` (0.01) is the fragrance multiplier, and ``a`` (0.1) is the fitness exponent.
            ``p`` (0.8) is the probability of moving toward the best butterfly rather than moving locally.
            Compilation initializes the per-agent ``fragrance`` buffer to zero.

        """

        super(BOA, self).__init__()

        self.c = 0.01
        self.a = 0.1
        self.p = 0.8

        self.build(params)

    def compile(self, space: Space) -> None:
        self.fragrance = np.zeros(space.n_agents)

    def _best_movement(
        self,
        agent_position: np.ndarray,
        best_position: np.ndarray,
        fragrance: np.ndarray,
        random: float,
    ) -> np.ndarray:
        new_position = agent_position + (random**2 * best_position - agent_position) * fragrance

        return new_position

    def _local_movement(
        self,
        agent_position: np.ndarray,
        j_position: np.ndarray,
        k_position: np.ndarray,
        fragrance: np.ndarray,
        random: float,
    ) -> np.ndarray:
        new_position = agent_position + (random**2 * j_position - k_position) * fragrance

        return new_position

    def update(self, space: Space) -> None:
        for i, agent in enumerate(space.agents):
            # Calculates fragrance for current agent (eq. 1)
            self.fragrance[i] = self.c * agent.fit**self.a

        for i, agent in enumerate(space.agents):
            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 < self.p:
                # Moves current agent towards the best one (eq. 2)
                agent.position = self._best_movement(agent.position, space.best_agent.position, self.fragrance[i], r1)
            else:
                j = np.random.randint(0, len(space.agents), None)
                k = r.integer(0, len(space.agents), exclude=j, size=None)

                # Moves current agent using a local movement (eq. 3)
                agent.position = self._local_movement(
                    agent.position,
                    space.agents[j].position,
                    space.agents[k].position,
                    self.fragrance[i],
                    r1,
                )

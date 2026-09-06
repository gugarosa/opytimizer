# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Flower Pollination Algorithm.

References:
    X.-S. Yang. Flower pollination algorithm for global optimization.
    International conference on unconventional computing and natural computation (2012).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class FPA(Optimizer):
    """Search through local pollination and global Lévy-flight pollination.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure flower pollination steps.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``beta`` (1.5) controls the Lévy distribution, and ``eta`` (0.2) scales global pollination.
            ``p`` (0.8) selects local pollination when the uniform draw does not exceed it.

        """

        super(FPA, self).__init__()

        self.beta = 1.5
        self.eta = 0.2
        self.p = 0.8

        self.build(params)

    def _global_pollination(self, agent_position: np.ndarray, best_position: np.ndarray) -> np.ndarray:
        step = d.generate_levy_distribution(self.beta)
        # Global pollination toward the best flower (eq. 1)
        global_pollination = self.eta * step * (best_position - agent_position)
        new_position = agent_position + global_pollination

        return new_position

    def _local_pollination(
        self,
        agent_position: np.ndarray,
        k_position: np.ndarray,
        l_position: np.ndarray,
        epsilon: float,
    ) -> np.ndarray:
        # Local differential pollination (eq. 3)
        local_pollination = epsilon * (k_position - l_position)
        new_position = agent_position + local_pollination

        return new_position

    def update(self, space: Space, function: Callable) -> None:
        for agent in space.agents:
            a = copy.deepcopy(agent)

            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 > self.p:
                a.position = self._global_pollination(agent.position, space.best_agent.position)
            else:
                epsilon = np.random.uniform(0.0, 1.0, 1)

                k = np.random.randint(0, len(space.agents), None)
                l = r.integer(0, len(space.agents), exclude=k, size=None)

                a.position = self._local_pollination(
                    agent.position,
                    space.agents[k].position,
                    space.agents[l].position,
                    epsilon,
                )
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Artificial Flora.

References:
    L. Cheng, W. Xue-han and Y. Wang. Artificial flora (AF) optimization algorithm.
    Applied Sciences (2018).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class AF(Optimizer):
    """Search with offspring dispersal and fitness-based flora selection.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure artificial flora propagation.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``c1`` (0.75) weights grandparent distance and ``c2`` (1.25) weights parent distance.
            ``m`` (10) is the number of offspring per agent, and ``Q`` (0.75) scales selection probability.
            Compilation samples per-agent ``p_distance`` and ``g_distance`` uniformly in ``[0, 1)``.

        """

        super(AF, self).__init__()

        self.c1 = 0.75
        self.c2 = 1.25

        self.m = 10

        self.Q = 0.75

        self.build(params)

    def compile(self, space: Space) -> None:
        self.p_distance = np.random.uniform(0.0, 1.0, space.n_agents)
        self.g_distance = np.random.uniform(0.0, 1.0, space.n_agents)

    def update(self, space: Space, function: Callable) -> None:
        space.agents.sort(key=lambda x: x.fit)
        new_agents = []

        for i, agent in enumerate(space.agents):
            for _ in range(self.m):
                a = copy.deepcopy(agent)

                r1 = np.random.uniform(0.0, 1.0, 1)
                r2 = np.random.uniform(0.0, 1.0, 1)
                r3 = np.random.uniform(0.0, 1.0, 1)

                # Calculates the new distance (eq. 1)
                distance = self.g_distance[i] * r1 * self.c1 + self.p_distance[i] * r2 * self.c2

                D = np.random.normal(0, distance, (space.n_variables, space.n_dimensions))

                # Updates offspring's position (eq. 5)
                a.position += D
                a.clip_by_bound()

                a.fit = function(a.position)

                # Calculates the probability of selection (eq. 6)
                p = np.fabs(np.sqrt(a.fit / space.agents[-1].fit)) * self.Q
                if r3 < p:
                    new_agents.append(a)

            # Updates both grandparent and parent distances (eq. 2 and 3)
            self.g_distance[i] = self.p_distance[i]
            self.p_distance[i] = np.std(agent.position - a.position)

        idx = np.random.choice(len(new_agents), space.n_agents, p=None, replace=False)
        space.agents = [new_agents[i] for i in idx]

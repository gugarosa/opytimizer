# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Brain Storm Optimization.

References:
    Y. Shi. Brain Storm Optimization Algorithm.
    International Conference in Swarm Intelligence (2011).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.math.general as g
import opytimizer.math.random as r
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class BSO(Optimizer):
    """Implement Brain Storm Optimization.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure idea clustering and recombination.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``m`` (5) is the number of idea clusters.
            ``p_replacement_cluster`` (0.2) is the probability of resampling a selected cluster's best idea.
            ``p_single_cluster`` (0.8) selects one-cluster rather than two-cluster recombination.
            ``p_single_best`` (0.4) selects the cluster's best idea in one-cluster recombination.
            ``p_double_best`` (0.5) selects both cluster leaders in two-cluster recombination.
            ``k`` (20) sets the width of the sigmoid perturbation schedule.

        """

        super(BSO, self).__init__()

        self.m = 5

        self.p_replacement_cluster = 0.2
        self.p_single_cluster = 0.8
        self.p_single_best = 0.4
        self.p_double_best = 0.5

        self.k = 20

        self.build(params)

    def _clusterize(self, agents: list[Agent]) -> tuple[np.ndarray, np.ndarray]:
        ideas = np.array([agent.position for agent in agents])
        labels = g.kmeans(ideas, self.m)

        ideas_idx_per_cluster, best_idx_per_cluster = [], []

        for i in range(self.m):
            ideas_idx = np.where(labels == i)[0]

            if len(ideas_idx) > 0:
                ideas_idx_per_cluster.append(ideas_idx)
            else:
                ideas_idx_per_cluster.append([])

            ideas_per_cluster = [(agents[j], j) for j in ideas_idx_per_cluster[i]]
            ideas_per_cluster.sort(key=lambda x: x[0].fit)

            if len(ideas_per_cluster) > 0:
                best_idx_per_cluster.append(ideas_per_cluster[0][1])
            else:
                best_idx_per_cluster.append(-1)

        return ideas_idx_per_cluster, best_idx_per_cluster

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def update(self, space: Space, function: Callable, iteration: int, n_iterations: int) -> None:
        ideas_idx_per_cluster, best_idx_per_cluster = self._clusterize(space.agents)

        r1 = np.random.uniform(0.0, 1.0, 1)
        if r1 < self.p_replacement_cluster:
            c = np.random.randint(0, self.m, None)
            space.agents[best_idx_per_cluster[c]].fill_with_uniform()

        for agent in space.agents:
            a = copy.deepcopy(agent)

            r2 = np.random.uniform(0.0, 1.0, 1)
            if r2 < self.p_single_cluster:
                c = np.random.randint(0, self.m, None)
                if len(ideas_idx_per_cluster[c]) > 0:
                    r3 = np.random.uniform(0.0, 1.0, 1)
                    if r3 < self.p_single_best:
                        a.position = copy.deepcopy(space.agents[best_idx_per_cluster[c]].position)
                    else:
                        j = np.random.randint(0, len(ideas_idx_per_cluster[c]), None)

                        a.position = copy.deepcopy(space.agents[ideas_idx_per_cluster[c][j]].position)
            else:
                if self.m > 1:
                    c1 = np.random.randint(0, self.m, None)
                    c2 = r.integer(0, self.m, exclude=c1, size=None)

                    if len(ideas_idx_per_cluster[c1]) > 0 and len(ideas_idx_per_cluster[c2]) > 0:
                        r4 = np.random.uniform(0.0, 1.0, 1)
                        if r4 < self.p_double_best:
                            a.position = (
                                space.agents[best_idx_per_cluster[c1]].position
                                + space.agents[best_idx_per_cluster[c2]].position
                            ) / 2
                        else:
                            u = np.random.randint(0, len(ideas_idx_per_cluster[c1]), None)
                            v = np.random.randint(0, len(ideas_idx_per_cluster[c2]), None)

                            a.position = (
                                space.agents[ideas_idx_per_cluster[c1][u]].position
                                + space.agents[ideas_idx_per_cluster[c2][v]].position
                            ) / 2

            r5 = np.random.uniform(0.0, 1.0, 1)
            csi = self._sigmoid((0.5 * n_iterations - iteration) / self.k) * r5

            a.position += csi * np.random.normal(0.0, 1.0, 1)
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

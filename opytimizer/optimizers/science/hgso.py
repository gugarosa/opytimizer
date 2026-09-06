# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Henry Gas Solubility Optimization.

References:
    F. Hashim et al. Henry gas solubility optimization: A novel physics-based algorithm.
    Future Generation Computer Systems (2019).

"""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class HGSO(Optimizer):
    """Clustered Henry-gas-solubility search with worst-agent replacement.

    """

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        """Configure clustering, gas scales, and movement weights.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``n_clusters`` (2) is the number of balanced, non-empty clusters and must
            be an integer between one and the population size when compiled.
            ``l1`` (0.0005), ``l2`` (100), and ``l3`` (0.001) scale initial Henry coefficients,
            gas pressures, and temperature constants, respectively.
            ``alpha`` (1.0) weights solubility-based attraction toward the global best.
            ``beta`` (1.0) scales fitness-dependent attraction and ``K`` (1.0) scales solubility.
            Compilation creates one ``coefficient`` and ``constant`` per cluster, plus a
            ``pressure`` array with one row per cluster and columns for the largest cluster.
            Changing the cluster count requires recompilation, while repeated runs otherwise retain state.

        """

        super().__init__()

        self.n_clusters = 2

        self.l1 = 0.0005
        self.l2 = 100
        self.l3 = 0.001

        self.alpha = 1.0
        self.beta = 1.0
        self.K = 1.0

        self.build(params)

    def compile(self, space: Space) -> None:
        if not isinstance(self.n_clusters, (int, np.integer)):
            raise TypeError("`n_clusters` must be an integer.")
        if not 1 <= self.n_clusters <= len(space.agents):
            raise ValueError("`n_clusters` must be between 1 and the population size.")

        n_agents_per_cluster = (len(space.agents) + self.n_clusters - 1) // self.n_clusters

        self.coefficient = self.l1 * np.random.uniform(0.0, 1.0, self.n_clusters)
        self.pressure = self.l2 * np.random.uniform(0.0, 1.0, (self.n_clusters, n_agents_per_cluster))
        self.constant = self.l3 * np.random.uniform(0.0, 1.0, self.n_clusters)

    def _update_position(self, agent: Agent, cluster_agent: Agent, best_agent: Agent, solubility: float) -> np.ndarray:
        # Updates the position of a single gas (eq. 10)
        gamma = self.beta * np.exp(-(best_agent.fit + 0.05) / (agent.fit + 0.05))
        flag = np.sign(np.random.uniform(-1, 1, 1))

        r1 = np.random.uniform(0.0, 1.0, 1)

        new_position = (
            agent.position
            + flag * r1 * gamma * (cluster_agent.position - agent.position)
            + flag * r1 * self.alpha * (solubility * best_agent.position - agent.position)
        )

        return new_position

    def update(self, space: Space, function: Callable, iteration: int, n_iterations: int) -> None:
        clusters = np.array_split(space.agents, self.pressure.shape[0])
        for i, cluster in enumerate(clusters):
            # Calculates the system's current temperature (eq. 8)
            T = np.exp(-iteration / n_iterations)

            # Updates Henry's coefficient (eq. 8)
            self.coefficient[i] *= np.exp(-self.constant[i] * (1 / T - 1 / 298.15))

            cluster = list(cluster)
            cluster.sort(key=lambda x: x.fit)

            for j, agent in enumerate(cluster):
                # Calculates agent's solubility (eq. 9)
                solubility = self.K * self.coefficient[i] * self.pressure[i][j]

                # Updates agent's position (eq. 10)
                agent.position = self._update_position(agent, cluster[0], space.best_agent, solubility)
                agent.clip_by_bound()

                agent.fit = function(agent.position)

        space.agents.sort(key=lambda x: x.fit)

        # Calculates the number of worst agents (eq. 11)
        r1 = np.random.uniform(0.0, 1.0)
        N = int(len(space.agents) * (r1 * (0.2 - 0.1) + 0.1))

        for agent in space.agents[len(space.agents) - N :]:
            # Updates bad agent's position (eq. 12)
            r2 = np.random.uniform(0.0, 1.0, 1)
            agent.position[:] = agent.lb[:, None] + r2 * (agent.ub - agent.lb)[:, None]
            agent.fit = function(agent.position)

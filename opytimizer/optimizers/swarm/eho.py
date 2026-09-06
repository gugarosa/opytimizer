# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Elephant Herding Optimization.

References:
    G.-G. Wang, S. Deb and L. Coelho. Elephant Herding Optimization.
    International Symposium on Computational and Business Intelligence (2015).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class EHO(Optimizer):
    """Search through clan-centered elephant movement and separation.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure elephant clan movement.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``alpha`` (0.5) weights attraction toward the clan leader, and ``beta`` (0.1) scales clan centers.
            ``n_clans`` (10) partitions agents into clans. Compilation sets ``n_ci`` to the integer
            quotient of population size and clan count, with remaining agents assigned to the final clan.

        """

        super(EHO, self).__init__()

        self.alpha = 0.5
        self.beta = 0.1

        self.n_clans = 10

        self.build(params)

    def compile(self, space: Space) -> None:
        self.n_ci = space.n_agents // self.n_clans

    def _get_agents_from_clan(self, agents: list[Agent], index: int) -> list[Agent]:
        start, end = index * self.n_ci, (index + 1) * self.n_ci

        if (index + 1) == self.n_clans:
            return sorted(agents[start:], key=lambda x: x.fit)

        return sorted(agents[start:end], key=lambda x: x.fit)

    def _updating_operator(self, agents: list[Agent], centers: np.ndarray, function: Callable) -> None:
        for i in range(self.n_clans):
            clan_agents = self._get_agents_from_clan(agents, i)
            for j, agent in enumerate(clan_agents):
                a = copy.deepcopy(agent)
                r1 = np.random.uniform(0.0, 1.0, 1)

                if j == 0:
                    # Updates its position (eq. 2)
                    a.position = self.beta * centers[i]
                else:
                    # Updates its position (eq. 1)
                    a.position += self.alpha * (clan_agents[0].position - a.position) * r1
                a.clip_by_bound()

                a.fit = function(a.position)
                if a.fit < agent.fit:
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

    def _separating_operator(self, agents: list[Agent]) -> None:
        for i in range(self.n_clans):
            clan_agents = self._get_agents_from_clan(agents, i)

            # Generates a new position for the worst agent in clan (eq. 4)
            worst = clan_agents[-1]
            worst.fill_with_uniform()

    def update(self, space: Space, function: Callable) -> None:
        centers = []

        for i in range(self.n_clans):
            clan_agents = self._get_agents_from_clan(space.agents, i)

            clan_center = np.mean(np.array([agent.position for agent in clan_agents]), axis=0)

            centers.append(clan_center)

        self._updating_operator(space.agents, centers, function)
        self._separating_operator(space.agents)

# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Germinal Center Optimization.

References:
    C. Villaseñor et al. Germinal center optimization algorithm.
    International Journal of Computational Intelligence Systems (2018).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class GCO(Optimizer):
    """Implement Germinal Center Optimization.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure germinal-center mutation.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``CR`` (0.7) is the per-variable mutation probability and ``F`` (1.25)
            scales the difference between donor cells.
            Compilation initializes cell lifetimes to 70 and duplication counters to one.

        """

        super(GCO, self).__init__()

        self.CR = 0.7
        self.F = 1.25

        self.build(params)

    def compile(self, space: Space) -> None:
        self.life = np.random.uniform(70, 70, space.n_agents)
        self.counter = np.ones(space.n_agents)

    def _mutate_cell(self, agent: Agent, alpha: Agent, beta: Agent, gamma: Agent) -> Agent:
        # Mutates a new cell based on distinct cells (alg. 2)
        a = copy.deepcopy(agent)

        for j in range(a.n_variables):
            r2 = np.random.uniform(0.0, 1.0, 1)
            if r2 < self.CR:
                a.position[j] = alpha.position[j] + self.F * (beta.position[j] - gamma.position[j])

        return a

    def _dark_zone(self, agents: list[Agent], function: Callable) -> None:
        # Performs the dark-zone update process (alg. 1)
        for i, agent in enumerate(agents):
            r1 = np.random.uniform(0, 100, 1)
            if r1 < self.life[i]:
                self.counter[i] += 1
            else:
                self.counter[i] = 1

            C = np.random.choice(len(agents), 3, p=self.counter / np.sum(self.counter), replace=False)

            a = self._mutate_cell(agent, agents[C[0]], agents[C[1]], agents[C[2]])
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

                self.life[i] += 10

    def _light_zone(self, agents: list[Agent]) -> None:
        # Performs the light-zone update process (alg. 1)
        fits = [agent.fit for agent in agents]
        min_fit, max_fit = np.min(fits), np.max(fits)

        for i, agent in enumerate(agents):
            self.life[i] = 10
            life_fit = (agent.fit - max_fit) / (min_fit - max_fit + c.EPSILON)
            self.life[i] += 10 * life_fit

    def update(self, space: Space, function: Callable) -> None:
        self._dark_zone(space.agents, function)
        self._light_zone(space.agents)

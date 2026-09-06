# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Slime Mould Algorithm.

References:
    S. Li, H. Chen, M. Wang, A. A. Heidari, S. Mirjalili
    Slime mould algorithm: A new method for stochastic optimization.
    Future Generation Computer Systems (2020).

"""

from typing import Any

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class SMA(Optimizer):
    """Implement Slime Mould Algorithm.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure slime-mould random exploration.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``z`` (0.03) is the probability of uniformly resampling an agent instead of moving it.
            Compilation allocates zero weights with the same shape as the population positions.

        """

        super(SMA, self).__init__()

        self.z = 0.03

        self.build(params)

    def compile(self, space: Space) -> None:
        self.weight = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

    def _update_weight(self, agents: list[Agent]):
        # Updates the weight of slime mould (eq. 2.5)
        best, worst = agents[0].fit, agents[-1].fit

        n_agents = len(agents)

        for i in range(n_agents):

            r1 = np.random.uniform(0, 1, (agents[i].n_variables, agents[i].n_dimensions))

            if i <= int(n_agents / 2):
                self.weight[i] = 1 + r1 * np.log10((best - agents[i].fit) / ((best - worst) + c.EPSILON) + 1)
            else:
                self.weight[i] = 1 - r1 * np.log10((best - agents[i].fit) / ((best - worst) + c.EPSILON) + 1)

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        space.agents.sort(key=lambda x: x.fit)

        self._update_weight(space.agents)

        a = np.arctanh(-((iteration + 1) / (n_iterations + 1)) + 1)
        b = 1 - (iteration + 1) / (n_iterations + 1)

        for i, agent in enumerate(space.agents):

            r2 = np.random.uniform(0.0, 1.0, 1)

            if r2 < self.z:
                agent.fill_with_uniform()
            else:
                p = np.tanh(np.abs(agent.fit - space.agents[0].fit))
                vb = np.random.uniform(-a, a, 1)
                vc = np.random.uniform(-b, b, 1)

                r3 = np.random.uniform(0.0, 1.0, 1)

                if r3 < p:
                    k = np.random.randint(0, len(space.agents), None)
                    l = r.integer(0, len(space.agents), exclude=k, size=None)
                    agent.position = space.agents[0].position + vb * (
                        self.weight[i] * (space.agents[k].position - space.agents[l].position)
                    )
                else:
                    agent.position *= vc

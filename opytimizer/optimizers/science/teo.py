# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Thermal Exchange Optimization.

References:
    A. Kaveh and A. Dadras. A novel meta-heuristic optimization algorithm: Thermal exchange optimization.
    Advances in Engineering Software (2017).

"""

import copy
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class TEO(Optimizer):
    """Implement Thermal Exchange Optimization.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure thermal exchange and elite memory.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``c1`` and ``c2`` (both True) enable the constant and time-dependent environment terms.
            ``pro`` (0.05) is the probability of randomly resetting one variable.
            ``n_TM`` (4) limits the number of elite agents stored in ``TM`` (initially an empty list).
            Updates retain the memory between runs, and compilation copies agents into the environment.

        """

        super(TEO, self).__init__()

        self.c1 = True
        self.c2 = True

        self.pro = 0.05

        self.n_TM = 4
        self.TM = []

        self.build(params)

    def compile(self, space: Space) -> None:
        self.environment = copy.deepcopy(space.agents)

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        space.agents.sort(key=lambda x: x.fit)

        self.TM.append(copy.deepcopy(space.agents[0]))
        self.TM = self.TM[-self.n_TM :]

        space.agents = space.agents[: -len(self.TM)] + self.TM
        space.agents.sort(key=lambda x: x.fit)

        # Calculates the time (eq. 9)
        time = iteration / n_iterations

        for env in self.environment:
            # Updates the environment's position (eq. 10)
            r1 = np.random.uniform(0.0, 1.0, 1)
            env.position = 1 - (self.c1 + self.c2 * (1 - time)) * r1 * env.position

        for agent, env in zip(space.agents, self.environment):
            # Calculates the agent's beta value (eq. 8)
            beta = agent.fit / space.agents[-1].fit

            # Updates the agent's position (eq. 11)
            agent.position = env.position + (agent.position - env.position) * np.exp(-beta * time)

            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 < self.pro:
                idx = np.random.randint(0, agent.n_variables, None)

                # Resets its position (eq. 12)
                r2 = np.random.uniform(0.0, 1.0, 1)
                agent.position[idx] = agent.lb[idx] + r2 * (agent.ub[idx] - agent.lb[idx])

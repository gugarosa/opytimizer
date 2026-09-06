# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Artificial Butterfly Optimization.

References:
    X. Qi, Y. Zhu and H. Zhang. A new meta-heuristic butterfly-inspired algorithm.
    Journal of Computational Science (2017).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class ABO(Optimizer):
    """Search using sunspot and canopy butterfly flight modes.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure artificial butterfly search.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``sunspot_ratio`` (0.9) selects the fraction of ranked agents treated as sunspot butterflies.
            ``a`` (2.0) is the initial exploration coefficient, reduced linearly over iterations.

        """

        super(ABO, self).__init__()

        self.sunspot_ratio = 0.9
        self.a = 2.0

        self.build(params)

    def _flight_mode(self, agent: Agent, neighbour: Agent, function: Callable) -> tuple[Agent, bool]:
        j = np.random.randint(0, agent.n_variables, None)
        r1 = np.random.uniform(-1, 1, 1)

        temp = copy.deepcopy(agent)

        # Updates temporary agent's position (eq. 1)
        temp.position[j] = agent.position[j] + (agent.position[j] - neighbour.position[j]) * r1
        temp.clip_by_bound()

        temp.fit = function(temp.position)
        if temp.fit < agent.fit:
            return temp.position, temp.fit, True

        return agent.position, agent.fit, False

    def update(self, space: Space, function: Callable, iteration: int, n_iterations: int) -> None:
        space.agents.sort(key=lambda x: x.fit)

        n_sunspots = int(self.sunspot_ratio * len(space.agents))
        for agent in space.agents[:n_sunspots]:
            k = np.random.randint(0, len(space.agents), None)

            # Performs a flight mode using sunspot butterflies (eq. 1)
            agent.position, agent.fit, _ = self._flight_mode(agent, space.agents[k], function)

        for agent in space.agents[n_sunspots:]:
            k = np.random.randint(0, len(space.agents) - n_sunspots, None)

            # Performs a flight mode using canopy butterflies (eq. 1)
            agent.position, agent.fit, is_better = self._flight_mode(agent, space.agents[k], function)

            if not is_better:
                k = np.random.randint(0, len(space.agents), None)
                r1 = np.random.uniform(0.0, 1.0, 1)

                # Calculates `D` (eq. 4)
                D = np.fabs(2 * r1 * space.agents[k].position - agent.position)

                r2 = np.random.uniform(0.0, 1.0, 1)

                # Updates the agent's position (eq. 3)
                a = self.a - self.a * (iteration / n_iterations)
                agent.position = space.agents[k].position - 2 * a * r2 - a * D
                agent.clip_by_bound()

                agent.fit = function(agent.position)

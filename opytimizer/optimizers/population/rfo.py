# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Red Fox Optimization.

References:
    D. Polap and M. Woźniak. Red fox optimization algorithm.
    Expert Systems with Applications (2021).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class RFO(Optimizer):
    """Implement Red Fox Optimization.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure fox observation and habitat replacement.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``phi`` is the observation angle, initially sampled uniformly from ``[0, 2 * pi)``.
            ``theta`` is the radius used when ``phi`` is zero, initially sampled uniformly from ``[0, 1)``.
            ``p_replacement`` (0.05) is the fraction replaced during updates, with the count fixed at compilation.
            The two initial random draws still occur when their values are overridden.

        """

        super(RFO, self).__init__()

        self.phi = np.random.uniform(0, 2 * np.pi, 1)[0]
        self.theta = np.random.uniform(0.0, 1.0, 1)[0]
        self.p_replacement = 0.05

        self.build(params)

    def compile(self, space: Space) -> None:
        self.n_replacement = int(self.p_replacement * space.n_agents)

    def _rellocation(self, agent: Agent, best_agent: Agent, function: Callable) -> None:
        temp = copy.deepcopy(agent)

        # Calculates the square root of euclidean distance between agent and best agent (eq. 1)
        distance = np.sqrt(np.linalg.norm(temp.position - best_agent.position))

        # Calculates individual reallocation (eq. 2)
        alpha = np.random.uniform(0, distance, 1)
        temp.position += alpha * np.sign(best_agent.position - temp.position)
        temp.clip_by_bound()

        temp.fit = function(temp.position)
        if temp.fit < agent.fit:
            agent.position = copy.deepcopy(temp.position)
            agent.fit = copy.deepcopy(temp.fit)

    def _noticing(self, agent: Agent, function: Callable, alpha: float) -> None:
        mu = np.random.uniform(0.0, 1.0, 1)
        if mu > 0.75:
            if self.phi != 0:
                # Calculates fox observation radius (eq. 4 - top)
                radius = alpha * np.sin(self.phi) / self.phi
            else:
                # Calculates fox observation radius (eq. 4 - bottom)
                radius = self.theta

            phi = np.random.uniform(0, 2 * np.pi, agent.n_variables)

            for j in range(agent.n_variables):
                total_sum = 0

                for k in range(j):
                    total_sum += np.sin(phi[k])

                # Updates the corresponding position (eq. 5)
                agent.position[j] += alpha * radius * (total_sum + np.cos(phi[j]))
            agent.clip_by_bound()

            agent.fit = function(agent.position)

    def update(self, space: Space, function: Callable) -> None:
        alpha = np.random.uniform(0, 0.2, 1)

        for agent in space.agents:
            self._rellocation(agent, space.best_agent, function)
            self._noticing(agent, function, alpha)

        space.agents.sort(key=lambda x: x.fit)

        # Calculates the habitat's center and diameter (eq. 6 and 7)
        habitat_center = (space.agents[0].position + space.agents[1].position) / 2
        habitat_diameter = np.sqrt(np.linalg.norm(space.agents[0].position - space.agents[1].position))

        k = np.random.uniform(0.0, 1.0, 1)

        for agent in space.agents[-self.n_replacement :]:
            # If sampled number is bigger than 0.45 (eq. 8 - top)
            if k >= 0.45:
                agent.fill_with_uniform()
                agent.position += habitat_center + habitat_diameter / 2

            # If sampled number is smaller than 0.45 (eq. 8 - bottom)
            else:
                # Reproduces parents into a new position (eq. 9)
                agent.position = k * (space.agents[0].position + space.agents[1].position) / 2

            agent.clip_by_bound()

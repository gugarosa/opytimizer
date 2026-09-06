# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Artificial Ecosystem-based Optimization.

References:
    W. Zhao, L. Wang and Z. Zhang.
    Artificial ecosystem-based optimization: a novel nature-inspired meta-heuristic algorithm.
    Neural Computing and Applications (2019).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class AEO(Optimizer):
    """Implement Artificial Ecosystem-based Optimization.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize ecosystem production, consumption, and decomposition.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            This optimizer has no algorithm-specific configuration keys.

        """

        super(AEO, self).__init__()

        self.build(params)

    def _production(self, agent: Agent, best_agent: Agent, iteration: int, n_iterations: int) -> Agent:
        # Performs the producer update (eq. 1)
        a = copy.deepcopy(agent)

        # Calculates the alpha factor (eq. 2)
        alpha = (1 - iteration / n_iterations) * np.random.uniform(0.0, 1.0, 1)

        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            a.position[j] = (1 - alpha) * best_agent.position[j] + alpha * np.random.uniform(lb, ub, a.n_dimensions)

        return a

    def _herbivore_consumption(self, agent: Agent, producer: Agent, C: float) -> Agent:
        # Performs the consumption update by a herbivore (eq. 6)
        a = copy.deepcopy(agent)
        a.position += C * (agent.position - producer.position)

        return a

    def _omnivore_consumption(self, agent: Agent, producer: Agent, consumer: Agent, C: float) -> Agent:
        # Performs the consumption update by an omnivore (eq. 8)
        a = copy.deepcopy(agent)

        r2 = np.random.uniform(0.0, 1.0, 1)
        a.position += C * r2 * (a.position - producer.position) + (1 - r2) * (a.position - consumer.position)

        return a

    def _carnivore_consumption(self, agent: Agent, consumer: Agent, C: float) -> Agent:
        # Performs the consumption update by a carnivore (eq. 7)
        a = copy.deepcopy(agent)
        a.position += C * (a.position - consumer.position)

        return a

    def _update_composition(
        self,
        agents: list[Agent],
        best_agent: Agent,
        function: Callable,
        iteration: int,
        n_iterations: int,
    ) -> None:
        # Wraps production and consumption updates over all agents and variables (eq. 1-8)
        agents.sort(key=lambda x: x.fit, reverse=True)
        for i, agent in enumerate(agents):
            if i == 0:
                a = self._production(agent, best_agent, iteration, n_iterations)
            else:
                r1 = np.random.uniform(0.0, 1.0, 1)

                v1 = np.random.normal(0.0, 1.0, 1)
                v2 = np.random.normal(0.0, 1.0, 1)

                # Calculates the consumption factor (eq. 4)
                C = 0.5 * v1 / np.abs(v2)

                if r1 < 1 / 3:
                    a = self._herbivore_consumption(agent, agents[0], C)
                elif 1 / 3 <= r1 <= 2 / 3:
                    j = int(np.random.uniform(1, i))
                    a = self._omnivore_consumption(agent, agents[0], agents[j], C)
                else:
                    j = int(np.random.uniform(1, i))
                    a = self._carnivore_consumption(agent, agents[j], C)

            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

    def _update_decomposition(self, agents: list[Agent], best_agent: Agent, function: Callable) -> None:
        # Wraps decomposition updates over all agents and variables (eq. 9)
        for agent in agents:
            a = copy.deepcopy(agent)

            # Calculates the decomposition factor (eq. 10)
            D = 3 * np.random.normal(0.0, 1.0, 1)

            r3 = np.random.uniform(0.0, 1.0, 1)

            # First weight coefficient (eq. 11)
            e = r3 * int(np.random.uniform(1, 2)) - 1

            # Second weight coefficient (eq. 12)
            _h = 2 * r3 - 1

            a.position = best_agent.position + D * (e * best_agent.position - _h * agent.position)
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

    def update(self, space: Space, function: Callable, iteration: int, n_iterations: int) -> None:
        self._update_composition(space.agents, space.best_agent, function, iteration, n_iterations)
        self._update_decomposition(space.agents, space.best_agent, function)

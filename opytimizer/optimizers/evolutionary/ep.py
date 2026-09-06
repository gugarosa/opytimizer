# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Evolutionary Programming.

Compilation initializes per-agent mutation strategies from the search bounds.
Updates mutate parents using equation 5.1, adapt and clip strategies using equation 5.2,
and retain tournament winners from the combined parent and child population.

References:
    A. E. Eiben and J. E. Smith. Introduction to Evolutionary Computing.
    Natural Computing Series (2013).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class EP(Optimizer):
    """Optimize a population with adaptive mutation and tournament selection.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize tournament size and strategy clipping.

        Args:
            params: Overrides for the supported optimizer parameters.

        Notes:
            Supported keys are ``bout_size`` (opponent count as a population fraction, 0.1)
            and ``clip_ratio`` (scale applied after clipping strategies to bounds, 0.05).

        """

        super(EP, self).__init__()

        self.bout_size = 0.1
        self.clip_ratio = 0.05

        self.build(params)

    def compile(self, space: Space) -> None:
        self.strategy = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

        for i in range(space.n_agents):
            for j, (lb, ub) in enumerate(zip(space.lb, space.ub)):
                self.strategy[i][j] = 0.05 * np.random.uniform(0, ub - lb, space.agents[i].n_dimensions)

    def _mutate_parent(self, agent: Agent, index: int, function: Callable) -> Agent:
        a = copy.deepcopy(agent)

        r1 = np.random.normal(0.0, 1.0, 1)

        a.position += self.strategy[index] * r1
        a.clip_by_bound()

        a.fit = function(a.position)

        return a

    def _update_strategy(self, index: int, lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
        n_variables, n_dimensions = self.strategy.shape[1], self.strategy.shape[2]

        r1 = np.random.normal(0.0, 1.0, (n_variables, n_dimensions))
        self.strategy[index] += r1 * (np.sqrt(np.abs(self.strategy[index])))

        for j, (lb, ub) in enumerate(zip(lower_bound, upper_bound)):
            self.strategy[index][j] = np.clip(self.strategy[index][j], lb, ub) * self.clip_ratio

    def update(self, space: Space, function: Callable) -> None:
        n_agents = len(space.agents)

        children = []
        for i, agent in enumerate(space.agents):
            a = self._mutate_parent(agent, i, function)
            self._update_strategy(i, agent.lb, agent.ub)

            children.append(a)

        space.agents += children

        n_individuals = int(n_agents * self.bout_size)
        wins = np.zeros(len(space.agents))

        for i, agent in enumerate(space.agents):
            for _ in range(n_individuals):
                index = np.random.randint(0, len(space.agents), None)
                if agent.fit < space.agents[index].fit:
                    wins[i] += 1

        space.agents = [agents for _, agents in sorted(zip(wins, space.agents), key=lambda pair: pair[0], reverse=True)]
        space.agents = space.agents[:n_agents]

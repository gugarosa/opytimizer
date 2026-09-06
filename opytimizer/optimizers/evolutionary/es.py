# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Evolution Strategies.

Compilation sets the offspring count and initializes mutation strategies from the bounds.
Updates mutate parents using equation 2, adapt strategies using equations 5-10,
and retain the fittest members of the combined parent and child population.

References:
    T. Bäck and H.–P. Schwefel. An Overview of Evolutionary Algorithms for Parameter Optimization.
    Evolutionary Computation (1993).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class ES(Optimizer):
    """Optimize a population with self-adaptive evolution strategies.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the offspring fraction.

        Args:
            params: Overrides for the supported optimizer parameters.

        Notes:
            The supported key is ``child_ratio`` (offspring count as a population fraction, 0.5).

        """

        super(ES, self).__init__()

        self.child_ratio = 0.5

        self.build(params)

    def compile(self, space: Space) -> None:
        self.n_children = int(space.n_agents * self.child_ratio)
        self.strategy = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

        for i in range(self.n_children):
            for j, (lb, ub) in enumerate(zip(space.lb, space.ub)):
                self.strategy[i][j] = 0.05 * np.random.uniform(0, ub - lb, space.agents[i].n_dimensions)

    def _mutate_parent(self, agent: Agent, index: int, function: Callable) -> Agent:
        a = copy.deepcopy(agent)

        r1 = np.random.normal(0.0, 1.0, 1)
        a.position += self.strategy[index] * r1
        a.clip_by_bound()

        a.fit = function(a.position)

        return a

    def _update_strategy(self, index: int) -> np.ndarray:
        n_variables, n_dimensions = self.strategy.shape[1], self.strategy.shape[2]

        tau = 1 / np.sqrt(2 * n_variables)
        tau_p = 1 / np.sqrt(2 * np.sqrt(n_variables))

        r1 = np.random.normal(0.0, 1.0, (n_variables, n_dimensions))
        r2 = np.random.normal(0.0, 1.0, (n_variables, n_dimensions))

        self.strategy[index] *= np.exp(tau_p * r1 + tau * r2)

    def update(self, space: Space, function: Callable) -> None:
        n_agents = len(space.agents)

        children = []
        for i in range(self.n_children):
            a = self._mutate_parent(space.agents[i], i, function)
            self._update_strategy(i)

            children.append(a)

        space.agents += children
        space.agents.sort(key=lambda x: x.fit)
        space.agents = space.agents[:n_agents]

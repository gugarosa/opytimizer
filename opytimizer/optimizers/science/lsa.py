# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Lightning Search Algorithm.

References:
    H. Shareef, A. Ibrahim and A. Mutlag. Lightning search algorithm.
    Applied Soft Computing (2015).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class LSA(Optimizer):
    """Implement Lightning Search Algorithm.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure lightning channel lifetime and energy.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``max_time`` (10) is the update interval for replacing the worst agent with the best.
            ``E`` (2.05) sets the energy offset and ``p_fork`` (0.01) is the probability
            of testing a random fork after a successful move.
            Compilation resets elapsed time and samples the shared direction array.

        """

        super(LSA, self).__init__()

        self.max_time = 10
        self.E = 2.05
        self.p_fork = 0.01

        self.build(params)

    def compile(self, space: Space) -> None:
        self.time = 0

        self.direction = np.sign(np.random.uniform(-1, 1, (space.n_variables, space.n_dimensions)))

    def _update_direction(self, agent: Agent, function: Callable) -> None:
        for j in range(agent.n_variables):
            direction = copy.deepcopy(agent)
            direction.position[j] += self.direction[j] * 0.005 * (agent.ub[j] - agent.lb[j])
            direction.clip_by_bound()

            direction.fit = function(direction.position)
            if direction.fit > agent.fit:
                self.direction[j] *= -1

    def _update_position(self, agent: Agent, best_agent: Agent, function: Callable, energy: float) -> None:
        a = copy.deepcopy(agent)

        distance = agent.position - best_agent.position

        for j in range(agent.n_variables):
            for k in range(agent.n_dimensions):
                if distance[j][k] == 0:
                    r1 = np.random.normal(0, energy)
                    a.position[j][k] += self.direction[j][k] * r1
                else:
                    if distance[j][k] < 0:
                        a.position[j][k] += np.random.exponential(np.fabs(distance[j][k]))
                    else:
                        a.position[j][k] -= np.random.exponential(distance[j][k])
        a.clip_by_bound()

        a.fit = function(a.position)
        if a.fit < agent.fit:
            agent.position = copy.deepcopy(a.position)
            agent.fit = copy.deepcopy(a.fit)

            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 < self.p_fork:
                a = copy.deepcopy(agent)
                a.fill_with_uniform()

                a.fit = function(a.position)
                if a.fit < agent.fit:
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

    def update(self, space: Space, function: Callable, iteration: int, n_iterations: int) -> None:
        self.time += 1
        if self.time >= self.max_time:
            space.agents.sort(key=lambda x: x.fit)
            space.agents[-1] = copy.deepcopy(space.agents[0])

            self.time = 0

        space.agents.sort(key=lambda x: x.fit)

        self._update_direction(space.agents[0], function)

        energy = self.E - 2 * np.exp(-5 * (n_iterations - iteration) / n_iterations)

        for agent in space.agents:
            self._update_position(agent, space.agents[0], function, energy)

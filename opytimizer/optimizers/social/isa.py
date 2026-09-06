# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Interactive Search Algorithm.

References:
    A. Mortazavi, V. Toğan and A. Nuhoğlu.
    Interactive search algorithm: A new hybrid metaheuristic optimization algorithm.
    Engineering Applications of Artificial Intelligence (2018).

"""

import copy
import time
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class ISA(Optimizer):
    """Implement Interactive Search Algorithm.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure interactive-search velocity and strategy selection.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``w`` (0.7) is velocity inertia and ``tau`` (0.3) is the probability of
            pairwise interaction instead of the global and weighted-position strategy.
            Compilation allocates zero local-best positions and velocities.
            Evaluation updates local and global bests only on strict improvement.

        """

        super(ISA, self).__init__()

        self.w = 0.7
        self.tau = 0.3

        self.build(params)

    def compile(self, space: Space) -> None:
        self.local_position = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))
        self.velocity = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

    def evaluate(self, space: Space, function: Callable) -> None:
        for i, agent in enumerate(space.agents):
            fit = function(agent.position)
            if fit < agent.fit:
                agent.fit = fit

                self.local_position[i] = copy.deepcopy(agent.position)

            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(self.local_position[i])
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: Space, function: Callable) -> None:
        space.agents.sort(key=lambda x: x.fit)
        best, worst = space.agents[0], space.agents[-1]

        coef = [(best.fit - agent.fit) / (best.fit - worst.fit + c.EPSILON) for agent in space.agents]
        w_coef = [cf / (np.sum(coef) + c.EPSILON) for cf in coef]

        w_position = np.sum([cf * agent.position for cf, agent in zip(w_coef, space.agents)], axis=0)
        w_fit = function(w_position)

        for i, agent in enumerate(space.agents):
            r1 = np.random.uniform(0.0, 1.0, 1)
            idx = r.integer(0, space.n_agents, exclude=i, size=None)

            if r1 >= self.tau:
                phi3 = np.random.uniform(0.0, 1.0, 1)
                phi2 = 2 * np.random.uniform(0.0, 1.0, 1)
                phi1 = -(phi2 + phi3) * np.random.uniform(0.0, 1.0, 1)

                # Updates the agent's velocity (eq. 6.1)
                self.velocity[i] = (
                    self.w * self.velocity[i]
                    + phi1 * (self.local_position[idx] - agent.position)
                    + phi2 * (space.best_agent.position - self.local_position[idx])
                    + phi3 * (w_position - self.local_position[idx])
                )
            else:
                r2 = np.random.uniform(0.0, 1.0, 1)
                if agent.fit < space.agents[idx].fit:
                    # Updates agent's velocity (eq. 6.2 - top)
                    self.velocity[i] = r2 * (agent.position - space.agents[idx].position)
                else:
                    # Updates agent's velocity (eq. 6.2 - bottom)
                    self.velocity[i] = r2 * (space.agents[idx].position - agent.position)

            # Updates agent's position and clip its bounds (eq. 6.3)
            agent.position += self.velocity[i]
            agent.clip_by_bound()

            agent.fit = function(agent.position)
            local_fit = function(self.local_position[i])

            if w_fit < agent.fit:
                if w_fit < local_fit:
                    self.local_position[i] = copy.deepcopy(w_position)
            else:
                if agent.fit < local_fit:
                    self.local_position[i] = copy.deepcopy(agent.position)

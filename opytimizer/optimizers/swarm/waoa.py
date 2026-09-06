# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Walrus Optimization Algorithm.

References:
    P. Trojovský and M. Dehghani. A new bio-inspired metaheuristic algorithm for
    solving optimization problems based on walruses behavior. Scientific Reports (2023).

"""

import copy
import time
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.math.random as r
from opytimizer.core.optimizer import Optimizer
from opytimizer.core.space import Space


class WAOA(Optimizer):
    """Search through walrus feeding, migration, and local exploration.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize walrus search.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            No algorithm-specific parameter defaults or compiled buffers are defined.
            Updates evaluate trial positions, while evaluation only promotes stored fitness to the global best.

        """

        super(WAOA, self).__init__()

        self.build(params)

    def evaluate(self, space: Space) -> None:
        for agent in space.agents:
            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: Space, function: Callable, iteration: int) -> None:
        for i, agent in enumerate(space.agents):
            a = copy.deepcopy(agent)

            r1 = np.random.randint(1, 3, (space.n_variables, space.n_dimensions))
            r2 = np.random.uniform(0.0, 1.0, (space.n_variables, space.n_dimensions))

            a.position = agent.position + r2 * (space.best_agent.position - r1 * agent.position)

            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

            k = r.integer(0, space.n_agents, exclude=i, size=None)

            if space.agents[k].fit < agent.fit:
                r3 = np.random.randint(1, 3, (space.n_variables, space.n_dimensions))
                r4 = np.random.uniform(0.0, 1.0, (space.n_variables, space.n_dimensions))

                a.position = agent.position + r4 * (space.agents[k].position - r3 * agent.position)
            else:
                r5 = np.random.uniform(0.0, 1.0, (space.n_variables, space.n_dimensions))

                a.position = agent.position + r5 * (agent.position - space.agents[k].position)

            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

            r6 = np.random.uniform(0.0, 1.0, (space.n_variables, space.n_dimensions))

            lb = (agent.lb / (iteration + 1)).reshape(-1, 1)
            ub = (agent.ub / (iteration + 1)).reshape(-1, 1)

            a.position = agent.position + (lb + (ub - r6 * lb))

            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

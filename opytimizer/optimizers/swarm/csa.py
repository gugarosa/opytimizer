# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Crow Search Algorithm.

References:
    A. Askarzadeh. A novel metaheuristic method for solving constrained engineering optimization problems:
    Crow search algorithm. Computers & Structures (2016).

"""

import copy
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class CSA(Optimizer):
    """Search by following remembered crow locations or relocating randomly.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure crow flight length and awareness.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``fl`` (2.0) scales flights toward another crow's memory, and ``AP`` (0.1) selects random relocation.
            Compilation zeros ``memory`` with shape ``(n_agents, n_variables, n_dimensions)``.
            Evaluation retains personal-best fitness and matching remembered positions.

        """

        super(CSA, self).__init__()

        self.fl = 2.0
        self.AP = 0.1

        self.build(params)

    def compile(self, space: Space) -> None:
        self.memory = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

    def evaluate(self, space: Space, function: Callable) -> None:
        for i, agent in enumerate(space.agents):
            fit = function(agent.position)
            if fit < agent.fit:
                agent.fit = fit

                # Updates the memory to current's agent position (eq. 5)
                self.memory[i] = copy.deepcopy(agent.position)

            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(self.memory[i])
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: Space) -> None:
        for agent in space.agents:
            r1 = np.random.uniform(0.0, 1.0, 1)
            r2 = np.random.uniform(0.0, 1.0, 1)

            j = np.random.randint(0, len(space.agents), None)

            if r1 >= self.AP:
                # Updates agent's position (eq. 2)
                agent.position += r2 * self.fl * (self.memory[j] - agent.position)
            else:
                agent.fill_with_uniform()

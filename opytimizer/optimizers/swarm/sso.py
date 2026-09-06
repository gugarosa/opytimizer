# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Simplified Swarm Optimization.

References:
    C. Bae et al. A new simplified swarm optimization (SSO) using exchange local search scheme.
    International Journal of Innovative Computing, Information and Control (2012).

"""

import copy
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class SSO(Optimizer):
    """Choose each swarm coordinate from current, personal-best, global-best, or random values.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure simplified swarm coordinate selection.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``C_w`` (0.1), ``C_p`` (0.4), and ``C_g`` (0.9) are cumulative thresholds for retaining a
            coordinate, copying its personal best, or copying its global best. Remaining draws sample ``[0, 1)``.
            Compilation zeros ``local_position`` with shape ``(n_agents, n_variables, n_dimensions)``.
            Evaluation retains personal-best fitness and its corresponding position.

        """

        super(SSO, self).__init__()

        self.C_w = 0.1
        self.C_p = 0.4
        self.C_g = 0.9

        self.build(params)

    def compile(self, space: Space) -> None:
        self.local_position = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

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

    def update(self, space: Space) -> None:
        for i, agent in enumerate(space.agents):
            for j in range(agent.n_variables):
                r1 = np.random.uniform(0.0, 1.0, 1)
                if r1 < self.C_w:
                    pass
                elif r1 < self.C_p:
                    agent.position[j] = self.local_position[i][j]
                elif r1 < self.C_g:
                    agent.position[j] = space.best_agent.position[j]
                else:
                    agent.position[j] = np.random.uniform(0.0, 1.0, agent.n_dimensions)

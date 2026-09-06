# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Satin Bowerbird Optimizer.

Movement, fitness-proportional selection, and Gaussian mutation follow equations 1-7 of the reference.

References:
    S. H. S. Moosavi and V. K. Bardsiri.
    Satin bowerbird optimizer: a new optimization algorithm to optimize ANFIS for software development effort estimation.
    Engineering Applications of Artificial Intelligence (2017).

"""

from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class SBO(Optimizer):
    """Search with fitness-weighted bower attraction and Gaussian mutation.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure bowerbird attraction and mutation.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``alpha`` (0.9) scales attraction, and ``p_mutation`` (0.05) is the per-variable mutation probability.
            ``z`` (0.02) scales each bounds' span into the compiled mutation standard deviations ``sigma``.

        """

        super(SBO, self).__init__()

        self.alpha = 0.9
        self.p_mutation = 0.05
        self.z = 0.02

        self.build(params)

    def compile(self, space: Space) -> None:
        self.sigma = [self.z * (ub - lb) for lb, ub in zip(space.lb, space.ub)]

    def update(self, space: Space, function: Callable) -> None:
        fitness = [1 / (1 + agent.fit) if agent.fit >= 0 else 1 + np.abs(agent.fit) for agent in space.agents]
        total_fitness = np.sum(fitness)
        probs = [fit / total_fitness for fit in fitness]

        for agent in space.agents:
            for j in range(agent.n_variables):
                s = np.random.choice(len(space.agents), 1, p=probs, replace=False)[0]

                lambda_k = self.alpha / (1 + probs[s])

                agent.position[j] += lambda_k * (
                    (space.agents[s].position[j] + space.best_agent.position[j]) / 2 - agent.position[j]
                )

                r1 = np.random.uniform(0.0, 1.0, 1)
                if r1 < self.p_mutation:
                    agent.position[j] += self.sigma[j] * np.random.normal(0.0, 1.0, 1)
            agent.clip_by_bound()

            agent.fit = function(agent.position)

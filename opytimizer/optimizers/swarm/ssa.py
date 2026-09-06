# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Salp Swarm Algorithm.

References:
    S. Mirjalili et al. Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems.
    Advances in Engineering Software (2017).

"""

from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class SSA(Optimizer):
    """Move a salp leader around the best position while followers track their predecessors.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize salp swarm search.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            No algorithm-specific parameter defaults or compiled buffers are defined.

        """

        super(SSA, self).__init__()

        self.build(params)

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        # Calculates the `c1` coefficient (eq. 3.2)
        c1 = 2 * np.exp(-((4 * iteration / n_iterations) ** 2))

        for i, _ in enumerate(space.agents):
            if i == 0:
                for j, (lb, ub) in enumerate(zip(space.agents[i].lb, space.agents[i].ub)):
                    c2 = np.random.uniform(0.0, 1.0, 1)
                    c3 = np.random.uniform(0.0, 1.0, 1)

                    if c3 < 0.5:
                        # Updates the leading salp position (eq. 3.1 - part 1)
                        space.agents[i].position[j] = space.best_agent.position[j] + c1 * ((ub - lb) * c2 + lb)
                    else:
                        # Updates the leading salp position (eq. 3.1 - part 2)
                        space.agents[i].position[j] = space.best_agent.position[j] - c1 * ((ub - lb) * c2 + lb)
            else:
                # Updates the follower salp position (eq. 3.4)
                space.agents[i].position = 0.5 * (space.agents[i].position + space.agents[i - 1].position)

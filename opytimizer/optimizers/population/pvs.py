# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Passing Vehicle Search.

Sampling peers without replacement preserves their distribution but changes
seeded trajectories relative to rejection sampling.

References:
    P. Savsani and V. Savsani. Passing vehicle search (PVS): A novel metaheuristic algorithm.
    Applied Mathematical Modelling (2016).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class PVS(Optimizer):
    """Implement Passing Vehicle Search.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize passing-vehicle search.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            This optimizer has no algorithm-specific configuration keys.
            Updating requires at least three agents to sample two distinct peers.

        """

        super(PVS, self).__init__()

        self.build(params)

    def update(self, space: Space, function: Callable) -> None:
        if space.n_agents < 3:
            raise ValueError("`n_agents` must be at least 3 for PVS.")

        space.agents.sort(key=lambda x: x.fit)
        for i, agent in enumerate(space.agents):
            a = copy.deepcopy(agent)

            R = np.random.choice(space.n_agents - 1, 2, replace=False)
            R += R >= i

            # Calculates the selected agents distances (eq. 16)
            D1 = 1 / space.n_agents * agent.fit
            D2 = 1 / space.n_agents * space.agents[R[0]].fit
            D3 = 1 / space.n_agents * space.agents[R[1]].fit

            # Calculates the selected agents velocities (eq. 17)
            V1 = np.random.uniform(0.0, 1.0, 1) * (1 - D1)
            V2 = np.random.uniform(0.0, 1.0, 1) * (1 - D2)
            V3 = np.random.uniform(0.0, 1.0, 1) * (1 - D3)

            # Calculates both `x` and `y` distance differences (eq. 18 and 19)
            x = np.fabs(D3 - D1)
            y = np.fabs(D3 - D2)

            # Calculates both `x1` and `y1` constraints (eq. 4 and 7)
            x1 = (V3 * x) / (V1 - V3)
            y1 = (V2 * x) / (V1 - V3)

            rnd = np.random.uniform(0.0, 1.0, 1)

            if V3 < V1:
                if (y - y1) > x1:
                    # Calculates the condition velocity (eq. 23)
                    Vco = V1 / (V1 - V3)

                    # Updates the temporary agent's position accordingly (eq. 20)
                    a.position += Vco * rnd * (a.position - space.agents[R[1]].position)
                else:
                    # Updates the temporary agent's position accordingly (eq. 21)
                    a.position += rnd * (a.position - space.agents[R[0]].position)
            else:
                # Updates the temporary agent's position accordingly (eq. 22)
                a.position += rnd * (space.agents[R[1]].position - a.position)

            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

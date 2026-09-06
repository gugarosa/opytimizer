# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Electro-Search Algorithm.

References:
    A. Tabari and A. Ahmad. A new optimization method: Electro-Search algorithm.
    Computers & Chemical Engineering (2017).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class ESA(Optimizer):
    """Implement Electro-Search Algorithm.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure electron candidates for electro-search.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``n_electrons`` (5) is the number of candidate electrons sampled per agent.
            Compilation initializes orbital-radius state with uniform draws from ``[0, 1)``.
            Rydberg and acceleration coefficients use uniform draws from ``[0, 1)``.
            Their sampling rationale is not documented in the original implementation.

        """

        super(ESA, self).__init__()

        self.n_electrons = 5

        self.build(params)

    def compile(self, space: Space) -> None:
        self.D = np.random.uniform(0.0, 1.0, (space.n_agents, space.n_variables, space.n_dimensions))

    def update(self, space: Space, function: Callable) -> None:
        for i, agent in enumerate(space.agents):
            a = copy.deepcopy(agent)

            electrons = [copy.deepcopy(agent) for _ in range(self.n_electrons)]
            for electron in electrons:
                r1 = np.random.uniform(0.0, 1.0, 1)
                n = np.random.randint(2, 6, None)

                # Updates the electron's position (eq. 3)
                electron.position += (2 * r1 - 1) * (1 - 1 / n**2) / self.D[i]
                electron.clip_by_bound()

                electron.fit = function(electron.position)

            electrons.sort(key=lambda x: x.fit)

            Re = np.random.uniform(0.0, 1.0, 1)
            Ac = np.random.uniform(0.0, 1.0, 1)

            # Updates the Orbital radius (eq. 4)
            self.D[i] = (electrons[0].position - space.best_agent.position) + Re * (
                1 / space.best_agent.position**2 - 1 / a.position**2
            )

            # Updates the temporary agent's position (eq. 5)
            a.position += Ac * self.D[i]
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

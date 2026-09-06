# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Simulated Annealing.

References:
    A. Khachaturyan, S. Semenovsovskaya and B. Vainshtein.
    The thermodynamic approach to the structure analysis of crystals.
    Acta Crystallographica (1981).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class SA(Optimizer):
    """Implement Simulated Annealing.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure annealing temperature and cooling.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``T`` (100) is the initial temperature used when accepting worse solutions.
            ``beta`` (0.999) multiplies ``T`` after each update, so temperature persists across runs.

        """

        super(SA, self).__init__()

        self.T = 100
        self.beta = 0.999

        self.build(params)

    def update(self, space: Space, function: Callable) -> None:
        for agent in space.agents:
            a = copy.deepcopy(agent)

            noise = np.random.normal(0, 0.1, (agent.n_variables, agent.n_dimensions))

            a.position += noise
            a.clip_by_bound()

            r1 = np.random.uniform(0.0, 1.0, 1)
            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)
            elif r1 < np.exp(-(a.fit - agent.fit) / self.T):
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

        self.T *= self.beta

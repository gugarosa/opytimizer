# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Electromagnetic Field Optimization.

References:
    H. Abedinpourshotorban et al.
    Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm.
    Swarm and Evolutionary Computation (2016).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class EFO(Optimizer):
    """Implement Electromagnetic Field Optimization.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure electromagnetic fields and replacement probabilities.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``positive_field`` (0.1) and ``negative_field`` (0.5) are the fractions of
            sorted agents assigned to the positive and negative fields.
            ``ps_ratio`` (0.1) is the probability of copying a positive-field coordinate.
            ``r_ratio`` (0.4) is the probability of randomly resetting one coordinate.
            ``phi`` (``(1 + sqrt(5)) / 2``) scales attraction relative to repulsion.
            ``RI`` (0) is the mutable coordinate index used for random replacement.

        """

        super(EFO, self).__init__()

        self.positive_field = 0.1
        self.negative_field = 0.5

        self.ps_ratio = 0.1
        self.r_ratio = 0.4
        self.phi = (1 + np.sqrt(5)) / 2

        self.RI = 0

        self.build(params)

    def _calculate_indexes(self, n_agents: int) -> tuple[int, int, int]:
        positive_index = int(np.random.uniform(0, n_agents * self.positive_field))

        negative_index = int(np.random.uniform(n_agents * (1 - self.negative_field), n_agents))

        neutral_index = int(np.random.uniform(n_agents * self.positive_field, n_agents * (1 - self.negative_field)))

        return positive_index, negative_index, neutral_index

    def update(self, space: Space, function: Callable) -> None:
        # Wraps Electromagnetic Field Optimization over all agents and variables (eq. 1-4)
        space.agents.sort(key=lambda x: x.fit)
        n_agents = len(space.agents)

        agent = copy.deepcopy(space.agents[0])
        force = np.random.uniform(0.0, 1.0, 1)

        for j in range(agent.n_variables):
            pos, neg, neu = self._calculate_indexes(n_agents)

            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 < self.ps_ratio:
                agent.position[j] = space.agents[pos].position[j]
            else:
                agent.position[j] = (
                    space.agents[neg].position[j]
                    + self.phi * force * (space.agents[pos].position[j] - space.agents[neu].position[j])
                    - force * (space.agents[neg].position[j] - space.agents[neu].position[j])
                )
        agent.clip_by_bound()

        r2 = np.random.uniform(0.0, 1.0, 1)
        if r2 < self.r_ratio:
            agent.position[self.RI] = np.random.uniform(agent.lb[self.RI], agent.ub[self.RI], 1)

            self.RI += 1
            if self.RI >= agent.n_variables:
                self.RI = 1

        agent.fit = function(agent.position)
        if agent.fit < space.agents[-1].fit:
            space.agents[-1] = copy.deepcopy(agent)

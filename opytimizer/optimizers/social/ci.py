# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Cohort Intelligence.

References:
    A. J. Kulkarni, I. P. Durugkar, M. Kumar. Cohort Intelligence: A Self Supervised Learning Behavior.
    IEEE International Conference on Systems, Man, and Cybernetics (2013).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.math.general as g
from opytimizer.core.optimizer import Optimizer
from opytimizer.core.space import Space


class CI(Optimizer):
    """Implement Cohort Intelligence.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure cohort sampling intervals and attempts.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``r`` (0.8) scales the bounds around a selected cohort member.
            ``t`` (3) is the number of candidate sampling attempts per agent per update.
            Compilation creates independent lower and upper sampling bounds for each agent.

        """

        super(CI, self).__init__()

        self.r = 0.8
        self.t = 3

        self.build(params)

    def compile(self, space: Space) -> None:
        lower = np.expand_dims(np.expand_dims(space.lb, -1), 0).astype(float)
        self.lower = np.repeat(lower, space.n_agents, axis=0)

        upper = np.expand_dims(np.expand_dims(space.ub, -1), 0).astype(float)
        self.upper = np.repeat(upper, space.n_agents, axis=0)

    def update(self, space: Space, function: Callable) -> None:
        fitness = [agent.fit for agent in space.agents]

        for i, agent in enumerate(space.agents):
            s = g.weighted_wheel_selection(fitness)

            self.lower[i] = space.agents[s].position - self.lower[i] * self.r / 2
            self.upper[i] = space.agents[s].position - self.upper[i] * self.r / 2

            for _ in range(self.t):
                a = copy.deepcopy(agent)

                for j, (lb, ub) in enumerate(zip(self.lower[i], self.upper[i])):
                    a.position[j] = np.random.uniform(lb, ub, agent.n_dimensions)
                a.clip_by_bound()

                a.fit = function(a.position)
                if a.fit < agent.fit:
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

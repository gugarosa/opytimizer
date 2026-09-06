# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Cuckoo Search.

References:
    X.-S. Yang and D. Suash. Cuckoo search via Lévy flights.
    World Congress on Nature & Biologically Inspired Computing (2009).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class CS(Optimizer):
    """Search nests using Lévy flights and differential replacement steps.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure cuckoo flight and nest replacement.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``alpha`` (1.0) scales Lévy steps and ``beta`` (1.5) controls their distribution.
            ``p`` (0.2) is the probability of retaining a nest during the abandonment step.
            The replacement mask is sampled with probability ``1 - p``.

        """

        super(CS, self).__init__()

        self.alpha = 1.0
        self.beta = 1.5
        self.p = 0.2

        self.build(params)

    def _generate_new_nests(self, agents: list[Agent], best_agent: Agent) -> list[Agent]:
        new_agents = copy.deepcopy(agents)
        for new_agent in new_agents:
            step = d.generate_levy_distribution(self.beta, new_agent.n_variables)
            step = np.expand_dims(step, axis=1)

            # Alpha scales the Lévy flight relative to the best nest (eq. 1)
            step_size = self.alpha * step * (new_agent.position - best_agent.position)

            g = np.random.normal(0.0, 1.0, new_agent.n_variables)
            g = np.expand_dims(g, axis=1)

            new_agent.position += step_size * g

        return new_agents

    def _generate_abandoned_nests(self, agents: list[Agent], prob: float) -> list[Agent]:
        new_agents = copy.deepcopy(agents)

        b = np.random.binomial(1, 1 - prob, len(agents))

        for j, new_agent in enumerate(new_agents):
            r1 = np.random.uniform(0.0, 1.0, 1)

            k = np.random.randint(0, len(agents) - 1, None)
            l = r.integer(0, len(agents) - 1, exclude=k, size=None)

            step_size = r1 * (agents[k].position - agents[l].position)
            new_agent.position += step_size * b[j]

        return new_agents

    def _evaluate_nests(self, agents: list[Agent], new_agents: list[Agent], function: Callable) -> None:
        for agent, new_agent in zip(agents, new_agents):
            new_agent.clip_by_bound()

            new_agent.fit = function(new_agent.position)
            if new_agent.fit < agent.fit:
                agent.position = copy.deepcopy(new_agent.position)
                agent.fit = copy.deepcopy(new_agent.fit)

    def update(self, space: Space, function: Callable) -> None:
        new_agents = self._generate_new_nests(space.agents, space.best_agent)
        self._evaluate_nests(space.agents, new_agents, function)

        new_agents = self._generate_abandoned_nests(space.agents, self.p)
        self._evaluate_nests(space.agents, new_agents, function)

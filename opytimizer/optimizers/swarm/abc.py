# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Artificial Bee Colony.

References:
    D. Karaboga and B. Basturk.
    A powerful and efficient algorithm for numerical function optimization: Artificial bee colony (ABC) algorithm.
    Journal of Global Optimization (2007).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class ABC(Optimizer):
    """Search food sources with employed, onlooker, and scout bees.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure artificial bee colony search.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``n_trials`` (10) is the failed-improvement threshold before scouting a food source.
            Compilation creates ``trial``, a zero-initialized failure counter for each agent.

        """

        super(ABC, self).__init__()

        self.n_trials = 10

        self.build(params)

    def compile(self, space: Space) -> None:
        self.trial = np.zeros(space.n_agents)

    def _evaluate_location(self, agent: Agent, neighbour: Agent, function: Callable, index: int) -> None:
        r1 = np.random.uniform(-1, 1, 1)

        a = copy.deepcopy(agent)

        # Change its location (eq. 2.2)
        a.position = agent.position + (agent.position - neighbour.position) * r1
        a.clip_by_bound()

        a.fit = function(a.position)
        if a.fit < agent.fit:
            self.trial[index] = 0

            agent.position = copy.deepcopy(a.position)
            agent.fit = copy.deepcopy(a.fit)
        else:
            self.trial[index] += 1

    def _send_employee(self, agents: list[Agent], function: Callable) -> None:
        for i, agent in enumerate(agents):
            source = np.random.randint(0, len(agents), None)
            self._evaluate_location(agent, agents[source], function, i)

    def _send_onlooker(self, agents: list[Agent], function: Callable) -> None:
        total = sum(agent.fit for agent in agents)

        k = 0
        while k < len(agents):
            for i, agent in enumerate(agents):
                r1 = np.random.uniform(0.0, 1.0, 1)
                # Food-source selection probability (eq. 2.1)
                probs = (agent.fit / (total + c.EPSILON)) + 0.1

                if r1 < probs:
                    k += 1

                    source = np.random.randint(0, len(agents), None)
                    self._evaluate_location(agent, agents[source], function, i)

    def _send_scout(self, agents: list[Agent], function: Callable) -> None:
        max_trial, max_index = np.max(self.trial), np.argmax(self.trial)
        if max_trial > self.n_trials:
            self.trial[max_index] = 0

            a = copy.deepcopy(agents[max_index])
            a.position += np.random.uniform(-1, 1, 1)
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agents[max_index].fit:
                agents[max_index] = copy.deepcopy(a)

    def update(self, space: Space, function: Callable) -> None:
        self._send_employee(space.agents, function)
        self._send_onlooker(space.agents, function)
        self._send_scout(space.agents, function)

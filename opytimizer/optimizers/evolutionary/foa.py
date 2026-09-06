# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Forest Optimization Algorithm.

Compilation initializes tree ages, and updates seed zero-aged trees locally, remove
old or excess trees, and seed a fraction of removed trees globally. The best tree's age
is reset after each update. Population limits are checked before updates and limiting.

References:
    M. Ghaemi, Mohammad-Reza F.-D. Forest Optimization Algorithm.
    Expert Systems with Applications (2014).

"""

import copy
from collections.abc import Callable
from numbers import Integral
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class FOA(Optimizer):
    """Optimize a population through local and global forest seeding.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize forest lifetime, population limits, and seeding parameters.

        Args:
            params: Overrides for the supported optimizer parameters.

        Notes:
            Supported keys are ``life_time`` (maximum tree age, 6), ``area_limit``
            (positive integral population limit, 30), ``LSC`` (local children per tree, 1),
            ``GSC`` (global seeding steps per candidate, 1), and ``transfer_rate``
            (fraction of removed trees selected for global seeding, 0.1).

        Raises:
            TypeError: The population limit is not an integer.
            ValueError: The population limit is not positive.

        """

        super(FOA, self).__init__()

        self.life_time = 6
        self.area_limit = 30
        self.LSC = 1
        self.GSC = 1
        self.transfer_rate = 0.1

        self.build(params)
        self._validate_area_limit()

    def _validate_area_limit(self) -> None:
        if not isinstance(self.area_limit, Integral):
            raise TypeError("`area_limit` must be an integer.")
        if self.area_limit <= 0:
            raise ValueError("`area_limit` must be positive.")

    def compile(self, space: Space) -> None:
        self.age = [0] * space.n_agents

    def _local_seeding(self, space: Space, function: Callable) -> None:
        new_agents = []
        for i, agent in enumerate(space.agents):
            if self.age[i] == 0:
                for _ in range(self.LSC):
                    child = copy.deepcopy(agent)

                    j = np.random.randint(0, child.n_variables, None)
                    child.position[j] += np.random.uniform(child.lb[j], child.ub[j], 1)
                    child.clip_by_bound()

                    child.fit = function(child.position)

                    new_agents.append(child)

        self.age = [age + 1 for age in self.age]

        space.agents += new_agents

        self.age += [0] * len(new_agents)

    def _population_limiting(self, space: Space) -> list[Agent]:
        self._validate_area_limit()

        candidate = []

        for i, _ in enumerate(space.agents):
            if self.age[i] > self.life_time:
                agent = space.agents.pop(i)
                self.age.pop(i)

                candidate.append(agent)

        space.agents, self.age = map(list, zip(*sorted(zip(space.agents, self.age), key=lambda x: x[0].fit)))

        if len(space.agents) > self.area_limit:
            candidate += space.agents[self.area_limit :]

            space.agents = space.agents[: self.area_limit]
            self.age = self.age[: self.area_limit]

        return candidate

    def _global_seeding(self, space: Space, function: Callable, candidate: list[Agent]) -> None:
        new_agents = []

        n_candidate = int(len(candidate) * self.transfer_rate)
        for agent in candidate[:n_candidate]:
            a = copy.deepcopy(agent)

            for _ in range(self.GSC):
                j = np.random.randint(0, a.n_variables, None)

                a.position[j] += np.random.uniform(a.lb[j], a.ub[j], 1)
                a.clip_by_bound()

                a.fit = function(a.position)

                new_agents.append(a)

        space.agents += new_agents

        self.age += [0] * len(new_agents)

    def update(self, space: Space, function: Callable) -> None:
        self._validate_area_limit()

        self._local_seeding(space, function)
        candidate = self._population_limiting(space)
        self._global_seeding(space, function, candidate)

        space.agents, self.age = map(list, zip(*sorted(zip(space.agents, self.age), key=lambda x: x[0].fit)))

        self.age[0] = 0

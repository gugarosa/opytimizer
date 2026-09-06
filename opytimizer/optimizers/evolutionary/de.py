# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Differential Evolution.

Each target requires three distinct other agents, so the population must contain at least
four agents. Updates use equations 1-4, including binomial mutation in equation 4.
Parameter domains are checked at construction and before updates, including after reassignment.

References:
    R. Storn. On the usage of differential evolution for function optimization.
    Proceedings of North American Fuzzy Information Processing (1996).

"""

import copy
from collections.abc import Callable, Mapping
from numbers import Real
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class DE(Optimizer):
    """Differential evolution with binomial crossover and greedy replacement.

    """

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        """Set and validate the crossover probability and differential weight.

        Args:
            params: Optional overrides for ``CR`` and ``F``.

        Notes:
            Supported keys are ``CR`` (crossover probability in ``[0, 1]``, 0.9)
            and ``F`` (differential weight in ``[0, 2]``, 0.7).

        Raises:
            TypeError: A crossover probability or differential weight is not real.
            ValueError: A crossover probability or differential weight is outside its domain.

        """

        super().__init__()

        self.CR = 0.9
        self.F = 0.7

        self.build(params)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if not isinstance(self.CR, Real):
            raise TypeError("`CR` should be a real number.")
        if not 0 <= self.CR <= 1:
            raise ValueError("`CR` should be between 0 and 1.")
        if not isinstance(self.F, Real):
            raise TypeError("`F` should be a real number.")
        if not 0 <= self.F <= 2:
            raise ValueError("`F` should be between 0 and 2.")

    def _mutate_agent(self, agent: Agent, alpha: Agent, beta: Agent, gamma: Agent) -> Agent:
        a = copy.deepcopy(agent)

        R = np.random.randint(0, agent.n_variables, None)

        for j in range(a.n_variables):
            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 < self.CR or j == R:
                a.position[j] = alpha.position[j] + self.F * (beta.position[j] - gamma.position[j])

        return a

    def update(self, space: Space, function: Callable) -> None:
        self._validate_parameters()

        for i, agent in enumerate(space.agents):
            C = np.random.choice(np.setdiff1d(range(0, len(space.agents)), i), 3, p=None, replace=False)

            a = self._mutate_agent(agent, space.agents[C[0]], space.agents[C[1]], space.agents[C[2]])
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

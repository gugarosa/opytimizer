# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Runner-Root Algorithm.

Updates explore through runner displacements, apply large and small root searches when
improvement stalls, and reinitialize the population after the configured stall limit.
Roulette selection uses a 0.1 regularizer when inverting fitness.

References:
    F. Merrikh-Bayat. The runner-root algorithm: A metaheuristic for solving unimodal and
    multimodal optimization problems inspired by runners and roots of plants in nature.
    Applied Soft Computing (2015).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class RRA(Optimizer):
    """Optimize a population using runner exploration and root searches.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize search scales and stagnation tracking.

        Args:
            params: Overrides for the supported optimizer parameters.

        Notes:
            Supported keys are ``d_runner`` (large-search scale, 2), ``d_root`` (small-search
            scale, 0.01), ``tol`` (relative improvement threshold, 0.01), ``max_stall``
            (stall count before restart, 1000), ``n_stall`` (initial stall count, 0), and
            ``last_best_fit`` (previous best fitness, ``FLOAT_MAX``, refreshed each update).

        """

        super(RRA, self).__init__()

        self.d_runner = 2
        self.d_root = 0.01
        self.tol = 0.01

        self.max_stall = 1000
        self.n_stall = 0

        self.last_best_fit = c.FLOAT_MAX

        self.build(params)

    def _stalling_search(
        self,
        daughters: list[Agent],
        function: Callable,
        is_large: bool = True,
    ) -> None:
        for _ in range(len(daughters) - 1):
            temp_daughter = copy.deepcopy(daughters[0])

            j = np.random.randint(0, temp_daughter.n_variables, None)

            if is_large:
                # Equation 4
                r1 = np.random.normal(0.0, 1.0, 1)
                temp_daughter.position[j] += self.d_runner * r1
            else:
                # Equation 5
                r1 = np.random.uniform(-0.5, 0.5, 1)
                temp_daughter.position[j] += self.d_root * r1

            temp_daughter.clip_by_bound()

            temp_daughter.fit = function(temp_daughter.position)
            if temp_daughter.fit < daughters[0].fit:
                daughters[0].position = copy.deepcopy(temp_daughter.position)
                daughters[0].fit = copy.deepcopy(temp_daughter.fit)

    def _roulette_selection(self, fitness: list[float], a: float = 0.1) -> int:
        min_fitness = np.min(fitness)

        # Equation 7 inverts fitness for minimization
        inv_fitness = [1 / (a + fit - min_fitness) for fit in fitness]
        total_fitness = np.sum(inv_fitness)

        # Equation 8
        probs = [fit / total_fitness for fit in inv_fitness]
        selected = np.random.choice(len(probs), 1, p=probs, replace=False)

        return selected[0]

    def update(self, space: Space, function: Callable) -> None:
        space.agents.sort(key=lambda x: x.fit)

        self.last_best_fit = space.agents[0].fit

        daughters = copy.deepcopy(space.agents)
        for daughter in daughters[1:]:
            r1 = np.random.uniform(-0.5, 0.5, 1)

            # Equation 2
            daughter.position += self.d_runner * r1
            daughter.clip_by_bound()

            daughter.fit = function(daughter.position)

        daughters.sort(key=lambda x: x.fit)

        # Equation 3
        effectiveness = np.fabs((self.last_best_fit - daughters[0].fit) / (self.last_best_fit + c.EPSILON))
        if effectiveness < self.tol:
            # Equation 4
            self._stalling_search(daughters, function, is_large=True)

            # Equation 5
            self._stalling_search(daughters, function, is_large=False)

        # Equation 6
        space.agents[0] = copy.deepcopy(daughters[0])

        daughters_fit = [daughter.fit for daughter in daughters]
        for agent in space.agents[1:]:
            idx = self._roulette_selection(daughters_fit)
            agent = copy.deepcopy(daughters[idx])

        # Equation 3
        effectiveness = np.fabs((self.last_best_fit - daughters[0].fit) / (self.last_best_fit + c.EPSILON))
        if effectiveness < self.tol:
            self.n_stall += 1
        else:
            self.n_stall = 0

        if self.n_stall == self.max_stall:
            for agent in space.agents:
                agent.fill_with_uniform()

            self.n_stall = 0

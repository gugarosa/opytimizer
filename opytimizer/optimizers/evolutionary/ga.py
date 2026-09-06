# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Genetic Algorithm.

Updates use roulette selection, arithmetic crossover, and Gaussian mutation as described
on page 8 of the reference, then retain the best parent and offspring solutions.

References:
    M. Mitchell. An introduction to genetic algorithms. MIT Press (1998).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.math.general as g
import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class GA(Optimizer):
    """Optimize a population with selection, crossover, and mutation.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize genetic selection and variation probabilities.

        Args:
            params: Overrides for the supported optimizer parameters.

        Notes:
            Supported keys are ``p_selection`` (selected population fraction, 0.75),
            ``p_mutation`` (per-variable mutation probability, 0.25), and
            ``p_crossover`` (pairwise crossover probability, 0.5).

        """

        super(GA, self).__init__()

        self.p_selection = 0.75
        self.p_mutation = 0.25
        self.p_crossover = 0.5

        self.build(params)

    def _roulette_selection(self, n_agents: int, fitness: list[float]) -> list[int]:
        n_individuals = int(n_agents * self.p_selection)
        if n_individuals % 2 != 0:
            n_individuals += 1

        max_fitness = np.max(fitness)

        # Invert fitness for minimization: f'(x) = f_max - f(x)
        inv_fitness = [max_fitness - fit + c.EPSILON for fit in fitness]
        total_fitness = np.sum(inv_fitness)

        probs = [fit / total_fitness for fit in inv_fitness]

        selected = np.random.choice(n_agents, n_individuals, p=probs, replace=False)

        return selected

    def _crossover(self, father: Agent, mother: Agent) -> tuple[Agent, Agent]:
        alpha, beta = copy.deepcopy(father), copy.deepcopy(mother)

        r1 = np.random.uniform(0.0, 1.0, 1)
        if r1 < self.p_crossover:
            r2 = np.random.uniform(0.0, 1.0, 1)

            alpha.position = r2 * father.position + (1 - r2) * mother.position
            beta.position = r2 * mother.position + (1 - r2) * father.position

        return alpha, beta

    def _mutation(self, alpha: Agent, beta: Agent) -> tuple[Agent, Agent]:
        for j in range(alpha.n_variables):
            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 < self.p_mutation:
                alpha.position[j] += np.random.normal(0.0, 1.0, 1)

            r2 = np.random.uniform(0.0, 1.0, 1)
            if r2 < self.p_mutation:
                beta.position[j] += np.random.normal(0.0, 1.0, 1)

        return alpha, beta

    def update(self, space: Space, function: Callable) -> None:
        new_agents = []
        n_agents = len(space.agents)

        fitness = [agent.fit + c.EPSILON for agent in space.agents]

        selected = self._roulette_selection(n_agents, fitness)
        for s in g.n_wise(selected):
            alpha, beta = self._crossover(space.agents[s[0]], space.agents[s[1]])
            alpha, beta = self._mutation(alpha, beta)

            alpha.clip_by_bound()
            beta.clip_by_bound()

            alpha.fit = function(alpha.position)
            beta.fit = function(beta.position)

            new_agents.extend([alpha, beta])

        space.agents += new_agents
        space.agents.sort(key=lambda x: x.fit)
        space.agents = space.agents[:n_agents]

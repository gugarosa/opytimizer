# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Sailfish Optimizer.

References:
    S. Shadravan, H. Naji and V. Bardsiri.
    The Sailfish Optimizer: A novel nature-inspired metaheuristic algorithm
    for solving constrained engineering optimization problems.
    Engineering Applications of Artificial Intelligence (2019).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class SFO(Optimizer):
    """Search through sailfish pursuit and sardine prey movement.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure sailfish prey density and attack power.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``PP`` (0.1) is the sailfish-to-sardine ratio used to allocate ``int(n_agents / PP)`` sardines.
            ``A`` (4) scales attack power, and ``e`` (0.001) controls its iteration-dependent change.
            Compilation creates the ``sardines`` population from randomized deep copies of the best agent.

        """

        super(SFO, self).__init__()

        self.PP = 0.1
        self.A = 4
        self.e = 0.001

        self.build(params)

    def compile(self, space: Space) -> None:
        self.sardines = [self._generate_random_agent(space.best_agent) for _ in range(int(space.n_agents / self.PP))]
        self.sardines.sort(key=lambda x: x.fit)

    def _generate_random_agent(self, agent: Agent) -> Agent:
        a = copy.deepcopy(agent)
        a.fill_with_uniform()

        return a

    def _calculate_lambda_i(self, n_sailfishes: int, n_sardines: int) -> float:
        # Calculates the prey density (eq. 8)
        PD = 1 - (n_sailfishes / (n_sailfishes + n_sardines))

        r1 = np.random.uniform(0.0, 1.0, 1)
        # Density-scaled pursuit coefficient (eq. 7)
        lambda_i = 2 * r1 * PD - PD

        return lambda_i

    def _update_sailfish(self, agent: Agent, best_agent: Agent, best_sardine: Agent, lambda_i: float) -> np.ndarray:
        r1 = np.random.uniform(0.0, 1.0, 1)
        # Sailfish pursuit of the best sardine (eq. 6)
        new_position = best_sardine.position - lambda_i * (
            r1 * (best_agent.position - best_sardine.position) / 2 - agent.position
        )

        return new_position

    def update(self, space: Space, function: Callable, iteration: int) -> None:
        best_sardine = self.sardines[0]

        n_sailfishes = len(space.agents)
        n_sardines = len(self.sardines)
        n_variables = space.agents[0].n_variables

        for agent in space.agents:
            lambda_i = self._calculate_lambda_i(n_sailfishes, n_sardines)

            agent.position = self._update_sailfish(agent, space.best_agent, best_sardine, lambda_i)
            agent.clip_by_bound()

            agent.fit = function(agent.position)

        # Calculates the attack power (eq. 10)
        AP = np.fabs(self.A * (1 - 2 * iteration * self.e))

        if AP < 0.5:
            # Calculates the number of sardines possible replacements (eq. 11)
            alpha = int(len(self.sardines) * AP)

            # Calculates the number of variables possible replacements (eq. 12)
            beta = int(n_variables * AP)

            selected_sardines = np.random.randint(0, n_sardines, alpha)

            for i in selected_sardines:
                selected_vars = np.random.randint(0, n_variables, beta)

                for j in selected_vars:
                    r1 = np.random.uniform(0.0, 1.0, 1)

                    # Updates the sardine's position (eq. 9)
                    self.sardines[i].position[j] = r1 * (
                        space.best_agent.position[j] - self.sardines[i].position[j] + AP
                    )
                self.sardines[i].clip_by_bound()

                self.sardines[i].fit = function(self.sardines[i].position)
        else:
            for sardine in self.sardines:
                # Updates the sardine's position (eq. 9)
                r1 = np.random.uniform(0.0, 1.0, 1)
                sardine.position = r1 * (space.best_agent.position - sardine.position + AP)
                sardine.clip_by_bound()

                sardine.fit = function(sardine.position)

        space.agents.sort(key=lambda x: x.fit)
        self.sardines.sort(key=lambda x: x.fit)

        for agent in space.agents:
            for sardine in self.sardines:
                # If agent is worse than sardine (eq. 13)
                if agent.fit > sardine.fit:
                    agent = copy.deepcopy(sardine)
                    break

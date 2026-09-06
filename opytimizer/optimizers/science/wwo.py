# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Water Wave Optimization.

References:
    Y.-J. Zheng. Water wave optimization: A new nature-inspired metaheuristic.
    Computers & Operations Research (2015).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class WWO(Optimizer):
    """Implement Water Wave Optimization.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure wave propagation, breaking, and refraction.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``h_max`` (5) is the maximum wave height, restored after successful propagation or refraction.
            ``alpha`` (1.001) controls wavelength adaptation and ``beta`` (0.001) scales breaking displacements.
            ``k_max`` (1) is the maximum number of coordinates tested during wave breaking.
            Compilation sets every height to ``h_max`` and every wavelength to 0.5.

        """

        super(WWO, self).__init__()

        self.h_max = 5

        self.alpha = 1.001
        self.beta = 0.001

        self.k_max = 1

        self.build(params)

    def compile(self, space: Space) -> None:
        self.height = np.random.uniform(self.h_max, self.h_max, space.n_agents)
        self.length = np.random.uniform(0.5, 0.5, space.n_agents)

    def _propagate_wave(self, agent: Agent, function: Callable, index: int) -> Agent:
        # Propagates wave into a new position (eq. 6)
        wave = copy.deepcopy(agent)

        for j in range(wave.n_variables):
            r1 = np.random.uniform(-1, 1, 1)
            wave.position[j] += r1 * self.length[index] * (j + 1)
        wave.clip_by_bound()

        wave.fit = function(wave.position)

        return wave

    def _refract_wave(self, agent: Agent, best_agent: Agent, function: Callable, index: int) -> tuple[float, float]:
        # Refract wave into a new position (eq. 8-9)
        current_fit = agent.fit

        for j in range(agent.n_variables):
            mean = (best_agent.position[j] + agent.position[j]) / 2
            std = np.fabs(best_agent.position[j] - agent.position[j]) / 2

            # Generates a new position (eq. 8)
            agent.position[j] = np.random.normal(mean, std, 1)
        agent.clip_by_bound()

        agent.fit = function(agent.position)

        # Re-calculates the new length (eq. 9)
        new_height = self.h_max
        new_length = self.length[index] * (current_fit / (agent.fit + c.EPSILON))

        return new_height, new_length

    def _break_wave(self, wave: Agent, function: Callable, j: int) -> Agent:
        # Breaks current wave into a new one (eq. 10)
        r1 = np.random.normal(0.0, 1.0, 1)

        broken_wave = copy.deepcopy(wave)
        broken_wave.position[j] += r1 * self.beta * (j + 1)
        broken_wave.clip_by_bound()

        broken_wave.fit = function(broken_wave.position)

        return broken_wave

    def _update_wave_length(self, agents: list[Agent]) -> None:
        agents.sort(key=lambda x: x.fit)

        for i, agent in enumerate(agents):
            self.length[i] *= self.alpha ** -(
                (agent.fit - agents[-1].fit + c.EPSILON) / (agents[0].fit - agents[-1].fit + c.EPSILON)
            )

    def update(self, space: Space, function: Callable) -> None:
        for i, agent in enumerate(space.agents):
            # Propagates a wave into a new temporary one (eq. 6)
            wave = self._propagate_wave(agent, function, i)
            if wave.fit < agent.fit:
                if wave.fit < space.best_agent.fit:
                    space.best_agent.position = copy.deepcopy(wave.position)
                    space.best_agent.fit = copy.deepcopy(wave.fit)

                    k = np.random.randint(1, self.k_max + 1, None)
                    for j in range(k):
                        # Breaks the propagated wave (eq. 10)
                        broken_wave = self._break_wave(wave, function, j)
                        if broken_wave.fit < space.best_agent.fit:
                            space.best_agent.position = copy.deepcopy(broken_wave.position)
                            space.best_agent.fit = copy.deepcopy(broken_wave.fit)

                agent.position = copy.deepcopy(wave.position)
                agent.fit = copy.deepcopy(wave.fit)

                self.height[i] = self.h_max
            else:
                self.height[i] -= 1

                if self.height[i] == 0:
                    # Refracts the wave and generates a new height and wave length (eq. 8-9)
                    self.height[i], self.length[i] = self._refract_wave(agent, space.best_agent, function, i)

        # Updates the wave length for all agents (eq. 7)
        self._update_wave_length(space.agents)

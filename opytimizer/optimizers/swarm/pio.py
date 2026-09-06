# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Pigeon-Inspired Optimization.

References:
    H. Duan and P. Qiao.
    Pigeon-inspired optimization: a new swarm intelligence optimizer for air robot path planning.
    International Journal of Intelligent Computing and Cybernetics (2014).

"""

from typing import Any

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class PIO(Optimizer):
    """Search using pigeon map-and-compass movement followed by landmark attraction.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure pigeon navigation phases.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``n_c1`` (150) ends the map-and-compass phase, and ``n_c2`` (200) ends landmark updates.
            ``R`` (0.2) controls exponential velocity decay.
            Compilation sets ``n_p`` to the population size and zeros the population-shaped ``velocity``.
            Landmark updates reduce ``n_p`` in place to select the leading pigeons used for the center.

        """

        super(PIO, self).__init__()

        self.n_c1 = 150
        self.n_c2 = 200

        self.R = 0.2

        self.build(params)

    def compile(self, space: Space) -> None:
        self.n_p = space.n_agents

        self.velocity = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

    def _calculate_center(self, agents: list[Agent]) -> np.ndarray:
        total_pos = np.zeros((agents[0].n_variables, agents[0].n_dimensions))
        total_fit = 0.0

        for agent in agents:
            total_pos += agent.position * agent.fit
            total_fit += agent.fit

        # Fitness-weighted landmark center (eq. 8)
        center = total_pos / (self.n_p * total_fit + c.EPSILON)

        return center

    def _update_center_position(self, position: np.ndarray, center: np.ndarray) -> None:
        r1 = np.random.uniform(0.0, 1.0, 1)
        # Landmark attraction (eq. 9)
        new_position = position + r1 * (center - position)

        return new_position

    def update(self, space: Space, iteration: int) -> None:
        if iteration < self.n_c1:
            for i, agent in enumerate(space.agents):
                # Updates current agent velocity (eq. 5)
                r1 = np.random.uniform(0.0, 1.0, 1)
                self.velocity[i] = self.velocity[i] * np.exp(-self.R * (iteration + 1)) + r1 * (
                    space.best_agent.position - agent.position
                )

                # Updates current agent position (eq. 6)
                agent.position += self.velocity[i]
        elif iteration < self.n_c2:
            # Calculates the number of possible pigeons (eq. 7)
            self.n_p = int(self.n_p / 2) + 1

            space.agents.sort(key=lambda x: x.fit)
            center = self._calculate_center(space.agents[: self.n_p])

            for agent in space.agents:
                agent.position = self._update_center_position(agent.position, center)

# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Manta Ray Foraging Optimization.

References:
    W. Zhao, Z. Zhang and L. Wang.
    Manta Ray Foraging Optimization: An effective bio-inspired optimizer for engineering applications.
    Engineering Applications of Artificial Intelligence (2020).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class MRFO(Optimizer):
    """Search through manta ray cyclone, chain, and somersault foraging.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure manta ray somersault movement.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``S`` (2.0) scales somersault displacement relative to the current and global-best positions.

        """

        super(MRFO, self).__init__()

        self.S = 2.0

        self.build(params)

    def _cyclone_foraging(
        self,
        agents: list[Agent],
        best_position: np.ndarray,
        i: int,
        iteration: int,
        n_iterations: int,
    ) -> np.ndarray:
        # Cyclone foraging balances random and best-position targets (eq. 3-7)
        r1 = np.random.uniform(0.0, 1.0, 1)
        r2 = np.random.uniform(0.0, 1.0, 1)
        r3 = np.random.uniform(0.0, 1.0, 1)

        beta = 2 * np.exp(r1 * (n_iterations - iteration + 1) / n_iterations) * np.sin(2 * np.pi * r1)

        if iteration / n_iterations < r2:
            r_position = np.zeros((agents[i].n_variables, agents[i].n_dimensions))

            for j, (lb, ub) in enumerate(zip(agents[i].lb, agents[i].ub)):
                r_position[j] = np.random.uniform(lb, ub, agents[i].n_dimensions)

            if i == 0:
                cyclone_foraging = (
                    r_position + r3 * (r_position - agents[i].position) + beta * (r_position - agents[i].position)
                )
            else:
                cyclone_foraging = (
                    r_position
                    + r3 * (agents[i - 1].position - agents[i].position)
                    + beta * (r_position - agents[i].position)
                )
        else:
            if i == 0:
                cyclone_foraging = (
                    best_position
                    + r3 * (best_position - agents[i].position)
                    + beta * (best_position - agents[i].position)
                )
            else:
                cyclone_foraging = (
                    best_position
                    + r3 * (agents[i - 1].position - agents[i].position)
                    + beta * (best_position - agents[i].position)
                )

        return cyclone_foraging

    def _chain_foraging(self, agents: list[Agent], best_position: np.ndarray, i: int) -> np.ndarray:
        # Chain foraging follows the preceding ray and global best (eq. 1-2)
        r1 = np.random.uniform(0.0, 1.0, 1)
        r2 = np.random.uniform(0.0, 1.0, 1)

        alpha = 2 * r1 * np.sqrt(np.abs(np.log(r1)))

        if i == 0:
            chain_foraging = (
                agents[i].position
                + r2 * (best_position - agents[i].position)
                + alpha * (best_position - agents[i].position)
            )
        else:
            chain_foraging = (
                agents[i].position
                + r2 * (agents[i - 1].position - agents[i].position)
                + alpha * (best_position - agents[i].position)
            )

        return chain_foraging

    def _somersault_foraging(self, position: np.ndarray, best_position: np.ndarray) -> np.ndarray:
        r1 = np.random.uniform(0.0, 1.0, 1)
        r2 = np.random.uniform(0.0, 1.0, 1)

        # Somersault displacement (eq. 8)
        somersault_foraging = position + self.S * (r1 * best_position - r2 * position)

        return somersault_foraging

    def update(self, space: Space, function: Callable, iteration: int, n_iterations: int) -> None:
        for i, agent in enumerate(space.agents):
            r1 = np.random.uniform(0.0, 1.0, 1)

            if r1 < 0.5:
                agent.position = self._cyclone_foraging(
                    space.agents, space.best_agent.position, i, iteration, n_iterations
                )
            else:
                agent.position = self._chain_foraging(space.agents, space.best_agent.position, i)
            agent.clip_by_bound()

            agent.fit = function(agent.position)
            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)

        for agent in space.agents:
            agent.position = self._somersault_foraging(agent.position, space.best_agent.position)

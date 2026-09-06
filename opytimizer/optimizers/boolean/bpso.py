# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Boolean Particle Swarm Optimization.

Compilation allocates Boolean velocity and personal-best arrays for the population.
Evaluation records improvements to personal and global best positions.

References:
    F. Afshinmanesh, A. Marandi and A. Rahimi-Kian.
    A Novel Binary Particle Swarm Optimization Method Using Artificial Immune System.
    IEEE International Conference on Smart Technologies (2005).

"""

import copy
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class BPSO(Optimizer):
    """Optimize Boolean variables with particle swarm updates.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the cognitive and social masks.

        Args:
            params: Overrides for the supported optimizer parameters.

        Notes:
            Supported keys are ``c1`` (cognitive mask, ``np.array([1])``) and
            ``c2`` (social mask, ``np.array([1])``).

        """

        super(BPSO, self).__init__()

        self.c1 = np.array([1])
        self.c2 = np.array([1])

        self.build(params)

    def compile(self, space: Space) -> None:
        self.local_position = np.zeros((space.n_agents, space.n_variables, space.n_dimensions), dtype=bool)
        self.velocity = np.zeros((space.n_agents, space.n_variables, space.n_dimensions), dtype=bool)

    def evaluate(self, space: Space, function: Callable) -> None:
        for i, agent in enumerate(space.agents):
            fit = function(agent.position)
            if fit < agent.fit:
                agent.fit = fit

                self.local_position[i] = copy.deepcopy(agent.position)

            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(self.local_position[i])
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: Space) -> None:
        for i, agent in enumerate(space.agents):
            r1 = np.random.randint(0, 2, agent.position.shape)
            r2 = np.random.randint(0, 2, agent.position.shape)

            local_partial = np.logical_and(
                self.c1,
                np.logical_xor(r1, np.logical_xor(self.local_position[i], agent.position)),
            )
            global_partial = np.logical_and(
                self.c2,
                np.logical_xor(r2, np.logical_xor(space.best_agent.position, agent.position)),
            )

            # Equation 1
            self.velocity[i] = np.logical_or(local_partial, global_partial)

            # Equation 2
            agent.position = np.logical_xor(agent.position, self.velocity[i])

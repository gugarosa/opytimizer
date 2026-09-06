# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Magnetic Optimization Algorithm.

References:
    M.-H. Tayarani and M.-R. Akbarzadeh. Magnetic-inspired optimization algorithms: Operators and structures.
    Swarm and Evolutionary Computation (2014).

"""

from typing import Any

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class MOA(Optimizer):
    """Implement Magnetic Optimization Algorithm.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure fitness-dependent magnetic mass.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``alpha`` (1.0) is the mass offset and ``rho`` (2.0) scales normalized fitness in the mass.
            Compilation requires a perfect-square population for the toroidal neighbor grid.

        """

        super(MOA, self).__init__()

        self.alpha = 1.0
        self.rho = 2.0

        self.build(params)

    def compile(self, space: Space) -> None:
        if not np.sqrt(space.n_agents).is_integer():
            raise ValueError("`n_agents` must be a perfect square.")

    def update(self, space: Space) -> None:
        space.agents.sort(key=lambda x: x.fit)

        # Gathers the best and worst agents and calculates a list of normalized fitness (eq. 2)
        best, worst = space.agents[0], space.agents[-1]
        fitness = [(agent.fit - best.fit) / (worst.fit - best.fit + c.EPSILON) for agent in space.agents]

        # Calculates the masses (eq. 3)
        mass = [self.alpha + self.rho * fit for fit in fitness]

        for i, agent in enumerate(space.agents):
            # Gathers the agents neighbours (eq. 4)
            root = np.sqrt(space.n_agents)
            north = int((i - root) % space.n_agents)
            south = int((i + root) % space.n_agents)
            west = int((i - 1) + ((i + root - 1) % root) // (root - 1) * root)
            east = int((i + 1) - (i % root) // (root - 1) * root)
            neighbours = [north, south, west, east]

            force = 0

            for n in neighbours:
                # Calculates the distance between current agent and neighbour (eq. 7)
                distance = np.linalg.norm(agent.position - space.agents[n].position)

                # Calculates the force between agents (eq. 5)
                force += (space.agents[n].position - agent.position) * fitness[n] / (distance + c.EPSILON)

            force = np.mean(force)

            # Updates the agent's velocity(eq. 9)
            r1 = np.random.uniform(0.0, 1.0, 1)
            velocity = force / mass[i] * r1

            # Updates the agent's position (eq. 10)
            agent.position += velocity
            agent.clip_by_bound()

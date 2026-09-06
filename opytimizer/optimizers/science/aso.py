# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Atom Search Optimization.

References:
    W. Zhao, L. Wang and Z. Zhang.
    A novel atom search optimization for dispersion coefficient estimation in groundwater.
    Future Generation Computer Systems (2019).

"""

from typing import Any

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class ASO(Optimizer):
    """Implement Atom Search Optimization.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure atom interaction and constraint weights.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``alpha`` (50.0) weights the interatomic potential and ``beta`` (0.2)
            weights attraction toward the best agent, divided by atom mass.
            Compilation allocates zero velocities for the population.

        """

        super(ASO, self).__init__()

        self.alpha = 50.0
        self.beta = 0.2

        self.build(params)

    def compile(self, space: Space) -> None:
        self.velocity = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

    def _calculate_mass(self, agents: list[Agent]) -> list[float]:
        # Calculates the atoms' masses (eq. 17 and 18)
        agents.sort(key=lambda x: x.fit)

        worst = agents[-1].fit
        best = agents[0].fit

        total_fit = np.sum([np.exp(-(agent.fit - best) / (worst - best + c.EPSILON)) for agent in agents])

        mass = [np.exp(-(agent.fit - best) / (worst - best + c.EPSILON)) / total_fit for agent in agents]

        return mass

    def _calculate_potential(
        self,
        agent: Agent,
        K_agent: Agent,
        average: np.ndarray,
        iteration: int,
        n_iterations: int,
    ) -> None:
        distance = np.linalg.norm(agent.position - average)
        radius = np.linalg.norm(agent.position - K_agent.position)

        rsmin = 1.1 + 0.1 * np.sin((iteration + 1) / n_iterations * np.pi / 2)
        rsmax = 1.24

        if radius / (distance + c.EPSILON) < rsmin:
            rs = rsmin
        else:
            if radius / (distance + c.EPSILON) > rsmax:
                rs = rsmax
            else:
                rs = radius / (distance + c.EPSILON)

        r1 = np.random.uniform(0.0, 1.0, 1)

        coef = (1 - iteration / n_iterations) ** 3
        potential = (
            coef
            * (12 * (-rs) ** (-13) - 6 * (-rs) ** (-7))
            * r1
            * ((K_agent.position - agent.position) / (radius + c.EPSILON))
        )

        return potential

    def _calculate_acceleration(
        self,
        agents: list[Agent],
        best_agent: Agent,
        mass: np.ndarray,
        iteration: int,
        n_iterations: int,
    ) -> np.ndarray:
        acceleration = np.zeros((len(agents), best_agent.n_variables, best_agent.n_dimensions))

        G = np.exp(-20.0 * iteration / n_iterations)

        K = int(len(agents) - (len(agents) - 2) * np.sqrt(iteration / n_iterations))
        K_agents, _ = map(list, zip(*sorted(zip(agents, mass), key=lambda x: x[1], reverse=True)[:K]))

        average = np.mean([agent.position for agent in K_agents])

        for i, agent in enumerate(agents):
            total_potential = np.zeros((agent.n_variables, agent.n_dimensions))

            for K_agent in K_agents:
                total_potential += self._calculate_potential(agent, K_agent, average, iteration, n_iterations)

            # Finally, calculates the acceleration (eq. 16)
            acceleration[i] = (
                G * self.alpha * total_potential + self.beta * (best_agent.position - agent.position) / mass[i]
            )

        return acceleration

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        # Calculates the masses (eq. 17 and 18)
        mass = self._calculate_mass(space.agents)

        # Calculates the acceleration (eq. 16)
        acceleration = self._calculate_acceleration(space.agents, space.best_agent, mass, iteration, n_iterations)

        for i, agent in enumerate(space.agents):
            # Updates current agent's velocity (eq. 21)
            r1 = np.random.uniform(0.0, 1.0, 1)
            self.velocity[i] = r1 * self.velocity[i] + acceleration[i]

            # Updates current agent's position (eq. 22)
            agent.position += self.velocity[i]

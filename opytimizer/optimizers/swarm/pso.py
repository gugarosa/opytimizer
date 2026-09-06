# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Particle Swarm Optimization-based algorithms.

Particle fitness stores the personal-best score rather than necessarily the score of the current position.
Its matching position is in ``local_position``. Compilation resets ``local_position`` and ``velocity`` to
zero arrays shaped ``(n_agents, n_variables, n_dimensions)``.

RPSO inherits PSO configuration and adds ``mass``, sampled uniformly from ``[0, 1)`` with the velocity shape.
Its update uses mass rather than ``w``. SAVPSO inherits PSO initialization and uses ``w`` but not ``c1`` or ``c2``.
VPSO adds ``v_velocity``, initialized to ones with the velocity shape at compilation.
AIWPSO initializes ``fitness`` with previous personal-best scores at iteration zero of each run.

References:
    J. Kennedy, R. C. Eberhart and Y. Shi. Swarm intelligence. Artificial Intelligence (2001).
    A. Nickabadi, M. M. Ebadzadeh and R. Safabakhsh.
    A novel particle swarm optimization algorithm with adaptive inertia weight. Applied Soft Computing (2011).
    M. Roder, G. H. de Rosa, L. A. Passos, A. L. D. Rossi and J. P. Papa.
    Harnessing Particle Swarm Optimization Through Relativistic Velocity.
    IEEE Congress on Evolutionary Computation (2020).
    H. Lu and W. Chen.
    Self-adaptive velocity particle swarm optimization for solving constrained optimization problems.
    Journal of global optimization (2008).
    W.-P. Yang. Vertical particle swarm optimization algorithm and its application in soft-sensor modeling.
    International Conference on Machine Learning and Cybernetics (2007).

"""

import copy
import time
from collections.abc import Callable, Mapping
from numbers import Real
from typing import Any

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


def _validate_nonnegative(name: str, value: Real) -> None:
    if not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real scalar.")
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"`{name}` must be finite and nonnegative.")


class PSO(Optimizer):
    """Particle swarm search with persistent velocities and personal bests.

    """

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        """Configure PSO coefficients before allocating particle state at compilation.

        Args:
            params: Overrides for PSO coefficients and any subclass parameters.

        Raises:
            TypeError: A coefficient is not a real scalar.
            ValueError: A coefficient is negative or nonfinite.

        Notes:
            ``w`` (0.7) weights the previous velocity, ``c1`` (1.7) attracts particles to their personal best,
            and ``c2`` (1.7) attracts them to the global best. All are finite, nonnegative real scalars.
            Inertia above one remains supported. Reassigned coefficients are checked before compilation,
            evaluation, and updates. State buffers and variant-specific behavior are described in this module.

        """

        super().__init__()

        self.w = 0.7
        self.c1 = 1.7
        self.c2 = 1.7

        self.build(params)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        for name in ("w", "c1", "c2"):
            _validate_nonnegative(name, getattr(self, name))

    def compile(self, space: Space) -> None:
        self._validate_parameters()

        self.local_position = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))
        self.velocity = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

    def evaluate(self, space: Space, function: Callable) -> None:
        self._validate_parameters()

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
        self._validate_parameters()

        for i, agent in enumerate(space.agents):
            r1 = np.random.uniform(0.0, 1.0, 1)
            r2 = np.random.uniform(0.0, 1.0, 1)

            # Updates agent's velocity (p. 294)
            self.velocity[i] = (
                self.w * self.velocity[i]
                + self.c1 * r1 * (self.local_position[i] - agent.position)
                + self.c2 * r2 * (space.best_agent.position - agent.position)
            )

            # Updates agent's position (p. 294)
            agent.position += self.velocity[i]


class AIWPSO(PSO):
    """Adapt PSO inertia from the fraction of particles improving their fitness.

    """

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        """Configure adaptive inertia limits and inherited PSO parameters.

        Args:
            params: Overrides for ``w_min``, ``w_max``, or inherited PSO parameters.

        Raises:
            TypeError: A coefficient or limit is not a real scalar.
            ValueError: A coefficient or limit is negative, nonfinite, or the limits are unordered.

        Notes:
            ``w_min`` (0.1) and ``w_max`` (0.9) bound the adapted inertia and require ``0 <= w_min <= w_max``.
            Equal limits and limits above one are supported. Both limits must be finite.
            ``w`` (0.7) sets the initial inertia before adaptation replaces it.
            ``c1`` (1.7) and ``c2`` (1.7) retain PSO's cognitive and social attraction meanings.

        """

        self.w_min = 0.1
        self.w_max = 0.9

        super().__init__(params)

    def _validate_parameters(self) -> None:
        super()._validate_parameters()
        for name in ("w_min", "w_max"):
            _validate_nonnegative(name, getattr(self, name))
        if self.w_max < self.w_min:
            raise ValueError("`w_max` must be greater than or equal to `w_min`.")

    def _compute_success(self, agents: list[Agent]) -> None:
        p = 0

        for i, agent in enumerate(agents):
            if agent.fit < self.fitness[i]:
                p += 1

            self.fitness[i] = agent.fit

        # Success-based inertia adaptation (eq. 16)
        self.w = (self.w_max - self.w_min) * (p / len(agents)) + self.w_min

    def update(self, space: Space, iteration: int) -> None:
        self._validate_parameters()

        if iteration == 0:
            self.fitness = [agent.fit for agent in space.agents]

        for i, agent in enumerate(space.agents):
            r1 = np.random.uniform(0.0, 1.0, 1)
            r2 = np.random.uniform(0.0, 1.0, 1)

            self.velocity[i] = (
                self.w * self.velocity[i]
                + self.c1 * r1 * (self.local_position[i] - agent.position)
                + self.c2 * r2 * (space.best_agent.position - agent.position)
            )

            agent.position += self.velocity[i]

        self._compute_success(space.agents)


class RPSO(PSO):
    """Mass-weighted PSO inspired by relativistic velocity.

    """

    def compile(self, space: Space) -> None:
        super().compile(space)
        self.mass = np.random.uniform(0.0, 1.0, (space.n_agents, space.n_variables, space.n_dimensions))

    def update(self, space: Space) -> None:
        self._validate_parameters()

        max_velocity = np.max(self.velocity)

        for i, agent in enumerate(space.agents):
            r1 = np.random.uniform(0.0, 1.0, 1)
            r2 = np.random.uniform(0.0, 1.0, 1)

            # Updates current agent velocity (eq. 11)
            gamma = 1 / np.sqrt(1 - (max_velocity**2 / c.LIGHT_SPEED**2))
            self.velocity[i] = (
                self.mass[i] * self.velocity[i] * gamma
                + self.c1 * r1 * (self.local_position[i] - agent.position)
                + self.c2 * r2 * (space.best_agent.position - agent.position)
            )

            agent.position += self.velocity[i]


class SAVPSO(PSO):
    """Self-adaptive velocity PSO with population-mean boundary corrections.

    """

    def update(self, space: Space) -> None:
        self._validate_parameters()

        positions = np.zeros((space.agents[0].position.shape[0], space.agents[0].position.shape[1]))

        for agent in space.agents:
            positions += agent.position
        positions /= len(space.agents)

        for i, agent in enumerate(space.agents):
            idx = np.random.randint(0, len(space.agents), None)

            # Updates current agent's velocity (eq. 8)
            r1 = np.random.uniform(0.0, 1.0, 1)
            self.velocity[i] = (
                self.w * np.fabs(self.local_position[idx] - self.local_position[i]) * np.sign(self.velocity[i])
                + r1 * (self.local_position[i] - agent.position)
                + (1 - r1) * (space.best_agent.position - agent.position)
            )

            agent.position += self.velocity[i]

            for j in range(agent.n_variables):
                r4 = np.random.uniform(0, 1, 1)

                if agent.position[j] > agent.ub[j]:
                    agent.position[j] = positions[j] + 1 * r4 * (agent.ub[j] - positions[j])

                if agent.position[j] < agent.lb[j]:
                    agent.position[j] = positions[j] + 1 * r4 * (agent.lb[j] - positions[j])


class VPSO(PSO):
    """PSO combining ordinary and vertical velocity components.

    """

    def compile(self, space: Space) -> None:
        super().compile(space)
        self.v_velocity = np.ones((space.n_agents, space.n_variables, space.n_dimensions))

    def update(self, space: Space) -> None:
        self._validate_parameters()

        for i, agent in enumerate(space.agents):
            r1 = np.random.uniform(0.0, 1.0, 1)
            r2 = np.random.uniform(0.0, 1.0, 1)

            # Updates current agent velocity (eq. 3)
            self.velocity[i] = (
                self.w * self.velocity[i]
                + self.c1 * r1 * (self.local_position[i] - agent.position)
                + self.c2 * r2 * (space.best_agent.position - agent.position)
            )

            # Updates current agent vertical velocity (eq. 4)
            self.v_velocity[i] -= (
                np.dot(self.velocity[i].T, self.v_velocity[i])
                / (np.dot(self.velocity[i].T, self.velocity[i]) + c.EPSILON)
            ) * self.velocity[i]

            # Updates current agent position (eq. 5)
            r1 = np.random.uniform(0.0, 1.0, 1)
            agent.position += r1 * self.velocity[i] + (1 - r1) * self.v_velocity[i]

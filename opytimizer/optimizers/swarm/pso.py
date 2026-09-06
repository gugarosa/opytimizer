"""Particle Swarm Optimization-based algorithms."""

import copy
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class PSO(Optimizer):
    """Particle swarm search with persistent velocities and personal bests.

    Attributes:
        w: Inertia weight multiplying the previous velocity. Defaults to ``0.7``.
        c1: Cognitive weight attracting a particle to its personal best. Defaults
            to ``1.7``.
        c2: Social weight attracting a particle to the global best. Defaults to
            ``1.7``.
        local_position: Personal-best positions, created by ``compile`` with shape
            ``(n_agents, n_variables, n_dimensions)``.
        velocity: Mutable velocity buffer with the same shape as ``local_position``.

    Particle fitness stores the personal-best score, which need not be the score
    of the current position. Its matching position is in ``local_position``.

    References:
        J. Kennedy, R. C. Eberhart and Y. Shi. Swarm intelligence.
        Artificial Intelligence (2001).

    """

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        """Configure PSO coefficients; allocate particle state later in ``compile``.

        Args:
            params: Optional overrides for ``w``, ``c1``, and ``c2``. Subclasses may
                define additional parameters, described in their own attributes.

        """

        super().__init__()

        self.w = 0.7
        self.c1 = 1.7
        self.c2 = 1.7

        self.build(params)

    def compile(self, space: Space) -> None:
        """Reset the shared personal-best and velocity buffers for this population.

        Args:
            space: A Space object containing meta-information.

        """

        self.local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def evaluate(self, space: Space, function: Callable) -> None:
        """Evaluate live positions and retain strict personal/global improvements.

        Args:
            space: A Space object that will be evaluated.
            function: Scalar objective receiving a particle's position array.

        """

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
        """Wraps Particle Swarm Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

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

    Attributes:
        w_min: Lower adaptation weight. Defaults to ``0.1``.
        w_max: Upper adaptation weight. Defaults to ``0.9``.
        fitness: Previous personal-best scores, initialized at iteration zero of
            each run.

    Inherited ``w`` sets the initial inertia; adaptation subsequently updates it.
    The cognitive and social coefficients retain the usual PSO meanings.

    References:
        A. Nickabadi, M. M. Ebadzadeh and R. Safabakhsh.
        A novel particle swarm optimization algorithm with adaptive inertia weight.
        Applied Soft Computing (2011).

    """

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        """Configure adaptive inertia limits and inherited PSO parameters.

        Args:
            params: Overrides for ``w_min``, ``w_max``, or inherited PSO parameters.

        """

        self.w_min = 0.1
        self.w_max = 0.9

        super().__init__(params)

    def _compute_success(self, agents: list[Agent]) -> None:
        """Computes the particles' success for updating inertia weight (eq. 16).

        Args:
            agents: List of agents.

        """

        p = 0

        for i, agent in enumerate(agents):
            if agent.fit < self.fitness[i]:
                p += 1

            self.fitness[i] = agent.fit

        self.w = (self.w_max - self.w_min) * (p / len(agents)) + self.w_min

    def update(self, space: Space, iteration: int) -> None:
        """Wraps Adaptive Inertia Weight Particle Swarm Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.

        """

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

    Configuration is inherited from :class:`PSO`.

    Attributes:
        mass: Per-component masses drawn uniformly from ``[0, 1)`` by ``compile``,
            with shape ``(n_agents, n_variables, n_dimensions)``.

    References:
        M. Roder, G. H. de Rosa, L. A. Passos, A. L. D. Rossi and J. P. Papa.
        Harnessing Particle Swarm Optimization Through Relativistic Velocity.
        IEEE Congress on Evolutionary Computation (2020).

    """

    def compile(self, space: Space) -> None:
        """Reset shared PSO state, then sample the additional mass buffer.

        Args:
            space: A Space object containing meta-information.

        """

        super().compile(space)
        self.mass = np.random.uniform(
            0.0, 1.0, (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def update(self, space: Space) -> None:
        """Wraps Relativistic Particle Swarm Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

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

    The constructor and state initialization are inherited from :class:`PSO`.
    This variant uses ``w`` but does not use the inherited ``c1``/``c2`` values in
    its update formula.

    References:
        H. Lu and W. Chen.
        Self-adaptive velocity particle swarm optimization for solving constrained optimization problems.
        Journal of global optimization (2008).

    """

    def update(self, space: Space) -> None:
        """Wraps Self-adaptive Velocity Particle Swarm Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        positions = np.zeros(
            (space.agents[0].position.shape[0], space.agents[0].position.shape[1])
        )

        for agent in space.agents:
            positions += agent.position
        positions /= len(space.agents)

        for i, agent in enumerate(space.agents):
            idx = np.random.randint(0, len(space.agents), None)

            # Updates current agent's velocity (eq. 8)
            r1 = np.random.uniform(0.0, 1.0, 1)
            self.velocity[i] = (
                self.w
                * np.fabs(self.local_position[idx] - self.local_position[i])
                * np.sign(self.velocity[i])
                + r1 * (self.local_position[i] - agent.position)
                + (1 - r1) * (space.best_agent.position - agent.position)
            )

            agent.position += self.velocity[i]

            for j in range(agent.n_variables):
                r4 = np.random.uniform(0, 1, 1)

                if agent.position[j] > agent.ub[j]:
                    agent.position[j] = positions[j] + 1 * r4 * (
                        agent.ub[j] - positions[j]
                    )

                if agent.position[j] < agent.lb[j]:
                    agent.position[j] = positions[j] + 1 * r4 * (
                        agent.lb[j] - positions[j]
                    )


class VPSO(PSO):
    """PSO combining ordinary and vertical velocity components.

    Configuration is inherited from :class:`PSO`.

    Attributes:
        v_velocity: Vertical velocity buffer initialized to ones by ``compile``.
            Its shape matches the shared PSO velocity buffer.

    References:
        W.-P. Yang. Vertical particle swarm optimization algorithm and its application in soft-sensor modeling.
        International Conference on Machine Learning and Cybernetics (2007).

    """

    def compile(self, space: Space) -> None:
        """Reset shared PSO state and initialize vertical velocities to ones.

        Args:
            space: A Space object containing meta-information.

        """

        super().compile(space)
        self.v_velocity = np.ones(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def update(self, space: Space) -> None:
        """Wraps Vertical Particle Swarm Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

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

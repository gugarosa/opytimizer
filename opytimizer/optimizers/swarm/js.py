# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Jellyfish Search-based algorithms.

Compilation replaces agent positions with a logistic-map sequence in the unit interval, without rescaling
to the search bounds. For ``x`` in ``[0, 1]``, ``x * (1 - x)`` lies in ``[0, 1/4]``.
Requiring ``0 < eta <= 4`` therefore keeps the recurrence in ``[0, 1]``.
Larger coefficients can leave this interval and diverge. This bound does not imply chaotic behavior
for every supported coefficient or starting point.

NBJS uses the same initialization and ocean current but omits the bounds' span from type A motion.

References:
    J.-S. Chou and D.-N. Truong. A novel metaheuristic optimizer inspired by behavior of jellyfish in ocean.
    Applied Mathematics and Computation (2020).
    NBJS: publication pending.

"""

from numbers import Real
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class JS(Optimizer):
    """Search with jellyfish ocean-current and local-motion dynamics.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure jellyfish initialization and motion coefficients.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Raises:
            TypeError: A coefficient is not a real scalar.
            ValueError: A coefficient is nonpositive, nonfinite, or ``eta`` exceeds four.

        Notes:
            ``eta`` (4.0) is the logistic-map coefficient and requires ``0 < eta <= 4``.
            ``beta`` (3.0) scales the population mean in the ocean current, and ``gamma`` (0.1) scales
            type A motion. Both require finite positive real values without an artificial upper bound.
            Coefficients are checked before initialization or movement can mutate agents.

        """

        super(JS, self).__init__()

        self.eta = 4.0
        self.beta = 3.0
        self.gamma = 0.1

        self.build(params)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        for name in ("eta", "beta", "gamma"):
            value = getattr(self, name)
            if not isinstance(value, Real):
                raise TypeError(f"`{name}` must be a real scalar.")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"`{name}` must be finite and positive.")
        if self.eta > 4:
            raise ValueError("`eta` must not exceed 4 to keep the logistic map in the unit interval.")

    def _initialize_chaotic_map(self, agents: list[Agent]) -> None:
        for i, agent in enumerate(agents):
            if i == 0:
                for j in range(agent.n_variables):
                    agent.position[j] = np.random.uniform(0.0, 1.0, agent.n_dimensions)
            else:
                for j in range(agent.n_variables):
                    # Calculates its position using logistic chaotic map (eq. 18)
                    agent.position[j] = self.eta * agents[i - 1].position[j] * (1 - agents[i - 1].position[j])

    def compile(self, space: Space) -> None:
        self._validate_parameters()
        self._initialize_chaotic_map(space.agents)

    def _ocean_current(self, agents: list[Agent], best_agent: Agent) -> np.ndarray:
        r1 = np.random.uniform(0.0, 1.0, 1)
        u = np.mean([agent.position for agent in agents])

        # Calculates the ocean current (eq. 9)
        trend = best_agent.position - self.beta * r1 * u

        return trend

    def _motion_a(self, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        # Type A motion scales with the bounds' span (eq. 12)
        r1 = np.random.uniform(0.0, 1.0, 1)
        motion = self.gamma * r1 * (np.expand_dims(ub, -1) - np.expand_dims(lb, -1))

        return motion

    def _motion_b(self, agent_i: Agent, agent_j: Agent) -> np.ndarray:
        r1 = np.random.uniform(0.0, 1.0, 1)

        if agent_i.fit >= agent_j.fit:
            # Determines its direction (eq. 15 - top)
            d = agent_j.position - agent_i.position
        else:
            # Determines its direction (eq. 15 - bottom)
            d = agent_i.position - agent_j.position

        motion = r1 * d

        return motion

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        self._validate_parameters()

        for agent in space.agents:
            r1 = np.random.uniform(0.0, 1.0, 1)

            # Calculates the time control mechanism (eq. 17)
            c = np.fabs((1 - iteration / n_iterations) * (2 * r1 - 1))

            if c >= 0.5:
                # Calculates the ocean current (eq. 9)
                trend = self._ocean_current(space.agents, space.best_agent)

                # Updates the location of current jellyfish (eq. 11)
                r2 = np.random.uniform(0.0, 1.0, 1)
                agent.position += r2 * trend
            else:
                r2 = np.random.uniform(0.0, 1.0, 1)
                if r2 > (1 - c):
                    # Update jellyfish's location with type A motion (eq. 12)
                    agent.position += self._motion_a(agent.lb, agent.ub)
                else:
                    # Updates jellyfish's location with type B motion (eq. 16)
                    j = np.random.randint(0, len(space.agents), None)
                    agent.position += self._motion_b(agent, space.agents[j])
            agent.clip_by_bound()


class NBJS(JS):
    """Apply jellyfish search with a bound-independent type A motion.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure the bound-independent jellyfish variant.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        Notes:
            ``eta`` (4.0) controls the logistic map, ``beta`` (3.0) scales the ocean current's population mean,
            and ``gamma`` (0.1) scales type A motion without multiplying by the bounds' span.
            The coefficient domains and validation behavior are inherited from :class:`JS`.

        """

        super(NBJS, self).__init__(params)

    def _motion_a(self, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        r1 = np.random.uniform(0.0, 1.0, 1)
        motion = self.gamma * r1

        return motion

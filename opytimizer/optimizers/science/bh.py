# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Black Hole.

References:
    A. Hatamlou. Black hole: A new heuristic optimization approach for data clustering.
    Information Sciences (2013).

"""

from collections.abc import Callable
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space
from opytimizer.utils import constant


class BH(Optimizer):
    """Implement Black Hole.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize black-hole attraction and event-horizon replacement.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            This optimizer has no algorithm-specific configuration keys.
            Stars crossing the fitness-derived event horizon are resampled within their bounds.

        """

        super(BH, self).__init__()

        self.build(params)

    def _update_position(self, agents: list[Agent], best_agent: Agent, function: Callable) -> float:
        # It updates every star position and calculates their event's horizon cost (eq. 3)
        cost = 0

        for agent in agents:
            r1 = np.random.uniform(0.0, 1.0, 1)
            agent.position += r1 * (best_agent.position - agent.position)
            agent.clip_by_bound()

            agent.fit = function(agent.position)
            if agent.fit < best_agent.fit:
                agent.position, best_agent.position = (
                    best_agent.position,
                    agent.position,
                )
                agent.fit, best_agent.fit = best_agent.fit, agent.fit

            cost += agent.fit

        return cost

    def _event_horizon(self, agents: list[Agent], best_agent: Agent, cost: float) -> None:
        # It calculates the stars' crossing an event horizon (eq. 4)
        radius = best_agent.fit / max(cost, constant.EPSILON)

        for agent in agents:
            distance = np.linalg.norm(best_agent.position - agent.position)
            if distance < radius:
                agent.fill_with_uniform()

    def update(self, space: Space, function: Callable) -> None:
        # Updates stars position and calculate their cost (eq. 3)
        cost = self._update_position(space.agents, space.best_agent, function)

        # Performs the Event Horizon (eq. 4)
        self._event_horizon(space.agents, space.best_agent, cost)

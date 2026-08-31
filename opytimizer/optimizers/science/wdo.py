"""Wind Driven Optimization."""

from typing import Any, Callable, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class WDO(Optimizer):
    """A WDO class, inherited from Optimizer.

    This is the designed class to define WDO-related
    variables and methods.

    References:
        Z. Bayraktar et al. The wind driven optimization technique and its application in electromagnetics.
        IEEE transactions on antennas and propagation (2013).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(WDO, self).__init__()

        self.v_max = 0.3
        self.alpha = 0.8
        self.g = 0.6
        self.c = 1.0
        self.RT = 1.5

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Wind Driven Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A function object.

        """

        for i, agent in enumerate(space.agents):
            index = np.random.randint(0, len(space.agents), None)

            # Updates velocity (eq. 15)
            self.velocity[i] = (
                (1 - self.alpha) * self.velocity[i]
                - self.g * agent.position
                + (
                    self.RT
                    * np.abs(1 / (index + 1) - 1)
                    * (space.best_agent.position - agent.position)
                )
                + (self.c * self.velocity[index] / (index + 1))
            )

            self.velocity = np.clip(self.velocity, -self.v_max, self.v_max)

            # Updates agent's position (eq. 16)
            agent.position += self.velocity[i]
            agent.clip_by_bound()

            agent.fit = function(agent.position)

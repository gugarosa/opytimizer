"""Firefly Algorithm."""

import copy
from typing import Any, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class FA(Optimizer):
    """A FA class, inherited from Optimizer.

    This is the designed class to define FA-related
    variables and methods.

    References:
        X.-S. Yang. Firefly algorithms for multimodal optimization.
        International symposium on stochastic algorithms (2009).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(FA, self).__init__()

        self.alpha = 0.5
        self.beta = 0.2
        self.gamma = 1.0

        self.build(params)

    def update(self, space: Space, n_iterations: int) -> None:
        """Wraps Firefly Algorithm over all agents and variables (eq. 3-9).

        Args:
            space: Space containing agents and update-related information.
            n_iterations: Maximum number of iterations.

        """

        delta = 1 - ((10e-4) / 0.9) ** (1 / n_iterations)
        self.alpha *= 1 - delta

        temp_agents = copy.deepcopy(space.agents)

        for agent in space.agents:
            for temp in temp_agents:
                # Distance is calculated by an euclidean distance between 'i' and 'j' (eq. 8)
                distance = np.linalg.norm(agent.position - temp.position)

                if agent.fit > temp.fit:
                    # Recalculate the attractiveness (eq. 6)
                    beta = self.beta * np.exp(-self.gamma * distance)

                    # Updates agent's position (eq. 9)
                    r1 = np.random.uniform(0.0, 1.0, 1)
                    agent.position = beta * (
                        temp.position + agent.position
                    ) + self.alpha * (r1 - 0.5)

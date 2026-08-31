"""Sine Cosine Algorithm."""

from typing import Any, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class SCA(Optimizer):
    """A SCA class, inherited from Optimizer.

    This is the designed class to define SCA-related
    variables and methods.

    References:
        S. Mirjalili. SCA: A Sine Cosine Algorithm for solving optimization problems.
        Knowledge-Based Systems (2016).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(SCA, self).__init__()

        self.r_min = 0
        self.r_max = 2

        self.a = 3

        self.build(params)

    def _update_position(
        self,
        agent_position: np.ndarray,
        best_position: np.ndarray,
        r1: float,
        r2: float,
        r3: float,
        r4: float,
    ) -> np.ndarray:
        """Updates a single particle position over a single variable (eq. 3.3).

        Args:
            agent_position: Agent's current position.
            best_position: Global best position.
            r1: Controls the next position's region.
            r2: Defines how far the movement should be.
            r3: Random weight for emphasizing or deemphasizing the movement.
            r4: Random number to decide whether sine or cosine should be used.

        Returns:
            (np.ndarray): A new position.

        """

        if r4 < 0.5:
            new_position = agent_position + r1 * np.sin(r2) * np.fabs(
                r3 * best_position - agent_position
            )

        else:
            new_position = agent_position + r1 * np.cos(r2) * np.fabs(
                r3 * best_position - agent_position
            )

        return new_position

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        """Wraps Sine Cosine Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Adaptively changing the r1 parameter, which controls the next position's region
        r1 = self.a - (iteration * self.a / n_iterations)

        # The r2 parameter defines how far the movement should be
        r2 = np.random.uniform(0, 2 * np.pi, 1)

        # A random weight for emphasizing or deemphasizing the movement
        r3 = np.random.uniform(self.r_min, self.r_max, 1)

        # A random number to decide whether sine or cosine should be used
        r4 = np.random.uniform(0.0, 1.0, 1)

        for agent in space.agents:
            agent.position = self._update_position(
                agent.position, space.best_agent.position, r1, r2, r3, r4
            )

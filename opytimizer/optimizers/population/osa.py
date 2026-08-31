"""Owl Search Algorithm."""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class OSA(Optimizer):
    """An OSA class, inherited from Optimizer.

    This is the designed class to define OSA-related
    variables and methods.

    References:
        M. Jain, S. Maurya, A. Rani and V. Singh.
        Owl search algorithm: A novelnature-inspired heuristic paradigm for global optimization.
        Journal of Intelligent & Fuzzy Systems (2018).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(OSA, self).__init__()

        self.beta = 1.9

        self.build(params)

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        """Wraps Owl Search Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        space.agents.sort(key=lambda x: x.fit)

        # Gathers best and worst agents (eq. 5 and 6)
        best = copy.deepcopy(space.agents[0])
        worst = copy.deepcopy(space.agents[-1])

        beta = self.beta - ((iteration + 1) / n_iterations) * self.beta

        for agent in space.agents:
            # Calculates the normalized intensity (eq. 4)
            intensity = (agent.fit - best.fit) / (worst.fit - best.fit + c.EPSILON)

            # Calculates the distance between owl and prey (eq. 7)
            distance = np.linalg.norm(agent.position - best.position)

            # Obtains the change in intensity (eq. 8)
            noise = np.random.uniform(0.0, 1.0, 1)
            intensity_change = intensity / (distance**2 + c.EPSILON) + noise

            p_vm = np.random.uniform(0.0, 1.0, 1)
            alpha = np.random.uniform(0.0, 0.5, 1)
            if p_vm < 0.5:
                # Updates current's owl position (eq. 9 - top)
                agent.position += (
                    beta
                    * intensity_change
                    * np.fabs(alpha * best.position - agent.position)
                )
            else:
                # Updates current's owl position (eq. 9 - bottom)
                agent.position -= (
                    beta
                    * intensity_change
                    * np.fabs(alpha * best.position - agent.position)
                )

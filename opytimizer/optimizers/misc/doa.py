"""Darcy Optimization Algorithm."""

from typing import Any, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class DOA(Optimizer):
    """A DOA class, inherited from Optimizer.

    This is the designed class to define DOA-related
    variables and methods.

    References:
        F. Demir et al. A survival classification method for hepatocellular carcinoma patients
        with chaotic Darcy optimization method based feature selection.
        Medical Hypotheses (2020).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(DOA, self).__init__()

        self.r = 1.0

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.chaotic_map = np.zeros((space.n_agents, space.n_variables))

    def _calculate_chaotic_map(self, lb: float, ub: float) -> float:
        """Calculates the chaotic map (eq. 3).

        Args:
            lb: Lower bound value.
            ub: Upper bound value.

        Returns:
            (float): A new value for the chaotic map.

        """

        r1 = np.random.uniform(lb, ub)

        # Calculates the chaotic map (eq. 3)
        c_map = self.r * r1 * (1 - r1) + ((4 - self.r) * np.sin(np.pi * r1)) / 4

        return c_map

    def update(self, space: Space) -> None:
        """Wraps Darcy Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        for i, agent in enumerate(space.agents):
            for j, (lb, ub) in enumerate(zip(agent.lb, agent.ub)):
                c_map = self._calculate_chaotic_map(lb, ub)

                # Updates the agent's position (eq. 6)
                agent.position[j] += (
                    (
                        2
                        * (space.best_agent.position[j] - agent.position[j])
                        / (c_map - self.chaotic_map[i][j])
                    )
                    * (ub - lb)
                    / len(space.agents)
                )

                self.chaotic_map[i][j] = c_map

                if (agent.position[j] < lb) or (agent.position[j] > ub):
                    # If yes, replace its value with the proposed equation (eq. 7)
                    agent.position[j] = space.best_agent.position[j] * c_map

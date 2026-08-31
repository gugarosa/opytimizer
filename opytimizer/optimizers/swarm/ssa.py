"""Salp Swarm Algorithm."""

from typing import Any, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class SSA(Optimizer):
    """A SSA class, inherited from Optimizer.

    This is the designed class to define SSA-related
    variables and methods.

    References:
        S. Mirjalili et al. Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems.
        Advances in Engineering Software (2017).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(SSA, self).__init__()

        self.build(params)

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        """Wraps Salp Swarm Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Calculates the `c1` coefficient (eq. 3.2)
        c1 = 2 * np.exp(-((4 * iteration / n_iterations) ** 2))

        for i, _ in enumerate(space.agents):
            if i == 0:
                for j, (lb, ub) in enumerate(
                    zip(space.agents[i].lb, space.agents[i].ub)
                ):
                    c2 = np.random.uniform(0.0, 1.0, 1)
                    c3 = np.random.uniform(0.0, 1.0, 1)

                    if c3 < 0.5:
                        # Updates the leading salp position (eq. 3.1 - part 1)
                        space.agents[i].position[j] = space.best_agent.position[
                            j
                        ] + c1 * ((ub - lb) * c2 + lb)
                    else:
                        # Updates the leading salp position (eq. 3.1 - part 2)
                        space.agents[i].position[j] = space.best_agent.position[
                            j
                        ] - c1 * ((ub - lb) * c2 + lb)
            else:
                # Updates the follower salp position (eq. 3.4)
                space.agents[i].position = 0.5 * (
                    space.agents[i].position + space.agents[i - 1].position
                )

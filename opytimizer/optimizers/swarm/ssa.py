"""Salp Swarm Algorithm.
"""

from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
from opytimizer.core import Optimizer
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


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

        logger.info("Overriding class: Optimizer -> SSA.")

        # Overrides its parent class with the receiving params
        super(SSA, self).__init__()

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        """Wraps Salp Swarm Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Calculates the `c1` coefficient (eq. 3.2)
        c1 = 2 * np.exp(-((4 * iteration / n_iterations) ** 2))

        # Iterates through every agent
        for i, _ in enumerate(space.agents):
            # Checks if it is the first agent
            if i == 0:
                # Iterates through every decision variable
                for j, (lb, ub) in enumerate(
                    zip(space.agents[i].lb, space.agents[i].ub)
                ):
                    # Generates two uniform random numbers
                    c2 = r.generate_uniform_random_number()
                    c3 = r.generate_uniform_random_number()

                    # Checks if random number is smaller than 0.5
                    if c3 < 0.5:
                        # Updates the leading salp position (eq. 3.1 - part 1)
                        space.agents[i].position[j] = space.best_agent.position[
                            j
                        ] + c1 * ((ub - lb) * c2 + lb)

                    # If random number is bigger or equal to 0.5
                    else:
                        # Updates the leading salp position (eq. 3.1 - part 2)
                        space.agents[i].position[j] = space.best_agent.position[
                            j
                        ] - c1 * ((ub - lb) * c2 + lb)

            # If it is not the first agent
            else:
                # Updates the follower salp position (eq. 3.4)
                space.agents[i].position = 0.5 * (
                    space.agents[i].position + space.agents[i - 1].position
                )

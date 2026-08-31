"""Boolean Particle Swarm Optimization."""

import copy
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class BPSO(Optimizer):
    """A BPSO class, inherited from Optimizer.

    This is the designed class to define boolean PSO-related
    variables and methods.

    References:
        F. Afshinmanesh, A. Marandi and A. Rahimi-Kian.
        A Novel Binary Particle Swarm Optimization Method Using Artificial Immune System.
        IEEE International Conference on Smart Technologies (2005).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(BPSO, self).__init__()

        self.c1 = np.array([1])
        self.c2 = np.array([1])

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions), dtype=bool
        )
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions), dtype=bool
        )

    def evaluate(self, space: Space, function: Callable) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A Space object that will be evaluated.
            function: A callable that will be used as the objective function.

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
        """Wraps Boolean Particle Swarm Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        for i, agent in enumerate(space.agents):
            r1 = np.random.randint(0, 2, agent.position.shape)
            r2 = np.random.randint(0, 2, agent.position.shape)

            local_partial = np.logical_and(
                self.c1,
                np.logical_xor(
                    r1, np.logical_xor(self.local_position[i], agent.position)
                ),
            )
            global_partial = np.logical_and(
                self.c2,
                np.logical_xor(
                    r2, np.logical_xor(space.best_agent.position, agent.position)
                ),
            )

            # Updates current agent velocities (eq. 1)
            self.velocity[i] = np.logical_or(local_partial, global_partial)

            # Updates current agent positions (eq. 2)
            agent.position = np.logical_xor(agent.position, self.velocity[i])

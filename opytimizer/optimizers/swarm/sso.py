"""Simplified Swarm Optimization."""

import copy
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class SSO(Optimizer):
    """A SSO class, inherited from Optimizer.

    This is the designed class to define SSO-related
    variables and methods.

    References:
        C. Bae et al. A new simplified swarm optimization (SSO) using exchange local search scheme.
        International Journal of Innovative Computing, Information and Control (2012).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(SSO, self).__init__()

        self.C_w = 0.1
        self.C_p = 0.4
        self.C_g = 0.9

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
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
        """Wraps Simplified Swarm Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        for i, agent in enumerate(space.agents):
            for j in range(agent.n_variables):
                r1 = np.random.uniform(0.0, 1.0, 1)
                if r1 < self.C_w:
                    pass
                elif r1 < self.C_p:
                    agent.position[j] = self.local_position[i][j]
                elif r1 < self.C_g:
                    agent.position[j] = space.best_agent.position[j]
                else:
                    agent.position[j] = np.random.uniform(0.0, 1.0, agent.n_dimensions)

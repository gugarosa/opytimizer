"""Artificial Flora."""

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class AF(Optimizer):
    """An AF class, inherited from Optimizer.

    This is the designed class to define AF-related
    variables and methods.

    References:
        L. Cheng, W. Xue-han and Y. Wang. Artificial flora (AF) optimization algorithm.
        Applied Sciences (2018).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(AF, self).__init__()

        self.c1 = 0.75
        self.c2 = 1.25

        self.m = 10

        self.Q = 0.75

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.p_distance = np.random.uniform(0.0, 1.0, space.n_agents)
        self.g_distance = np.random.uniform(0.0, 1.0, space.n_agents)

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Artificial Flora over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        space.agents.sort(key=lambda x: x.fit)
        new_agents = []

        for i, agent in enumerate(space.agents):
            for _ in range(self.m):
                a = copy.deepcopy(agent)

                r1 = np.random.uniform(0.0, 1.0, 1)
                r2 = np.random.uniform(0.0, 1.0, 1)
                r3 = np.random.uniform(0.0, 1.0, 1)

                # Calculates the new distance (eq. 1)
                distance = (
                    self.g_distance[i] * r1 * self.c1
                    + self.p_distance[i] * r2 * self.c2
                )

                D = np.random.normal(
                    0, distance, (space.n_variables, space.n_dimensions)
                )

                # Updates offspring's position (eq. 5)
                a.position += D
                a.clip_by_bound()

                a.fit = function(a.position)

                # Calculates the probability of selection (eq. 6)
                p = np.fabs(np.sqrt(a.fit / space.agents[-1].fit)) * self.Q
                if r3 < p:
                    new_agents.append(a)

            # Updates both grandparent and parent distances (eq. 2 and 3)
            self.g_distance[i] = self.p_distance[i]
            self.p_distance[i] = np.std(agent.position - a.position)

        idx = np.random.choice(len(new_agents), space.n_agents, p=None, replace=False)
        space.agents = [new_agents[i] for i in idx]

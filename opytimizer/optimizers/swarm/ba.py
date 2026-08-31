"""Bat Algorithm."""

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class BA(Optimizer):
    """A BA class, inherited from Optimizer.

    This is the designed class to define BA-related
    variables and methods.

    References:
        X.-S. Yang. A new metaheuristic bat-inspired algorithm.
        Nature inspired cooperative strategies for optimization (2010).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(BA, self).__init__()

        self.f_min = 0
        self.f_max = 2

        self.A = 0.5
        self.r = 0.5

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.frequency = np.random.uniform(self.f_min, self.f_max, space.n_agents)
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )
        self.loudness = np.random.uniform(0, self.A, space.n_agents)
        self.pulse_rate = np.random.uniform(0, self.r, space.n_agents)

    def update(self, space: Space, function: Callable, iteration: int) -> None:
        """Wraps Bat Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.
            iteration: Current iteration.

        """

        alpha = 0.9

        for i, agent in enumerate(space.agents):
            # Updates frequency (eq. 2)
            # Note that we have to apply (min - max) instead of (max - min) or it will not converge
            beta = np.random.uniform(0.0, 1.0, 1)
            self.frequency[i] = self.f_min + (self.f_min - self.f_max) * beta

            # Updates velocity (eq. 3)
            self.velocity[i] += (
                agent.position - space.best_agent.position
            ) * self.frequency[i]

            # Updates agent's position (eq. 4)
            agent.position += self.velocity[i]

            p = np.random.uniform(0.0, 1.0, 1)
            e = np.random.normal(0.0, 1.0, 1)
            if p > self.pulse_rate[i]:
                # Performs a local random walk (eq. 5)
                # We apply 0.001 to limit the step size
                agent.position = space.best_agent.position + 0.001 * e * np.mean(
                    self.loudness
                )
            agent.clip_by_bound()

            agent.fit = function(agent.position)
            if p < self.loudness[i] and agent.fit < space.best_agent.fit:
                space.best_agent = copy.deepcopy(agent)

                # Increasing pulse rate (eq. 6 - left)
                self.pulse_rate[i] = self.r * (1 - np.exp(-alpha * iteration))

                # Decreasing loudness (eq. 6 - right)
                self.loudness[i] = self.A * alpha

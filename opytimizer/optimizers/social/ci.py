"""Cohort Intelligence."""

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

import opytimizer.math.general as g
from opytimizer.core.optimizer import Optimizer
from opytimizer.core.space import Space


class CI(Optimizer):
    """A CI class, inherited from Optimizer.

    This is the designed class to define CI-related
    variables and methods.

    References:
        A. J. Kulkarni, I. P. Durugkar, M. Kumar. Cohort Intelligence: A Self Supervised Learning Behavior.
        IEEE International Conference on Systems, Man, and Cybernetics (2013).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(CI, self).__init__()

        self.r = 0.8
        self.t = 3

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        lower = np.expand_dims(np.expand_dims(space.lb, -1), 0).astype(float)
        self.lower = np.repeat(lower, space.n_agents, axis=0)

        upper = np.expand_dims(np.expand_dims(space.ub, -1), 0).astype(float)
        self.upper = np.repeat(upper, space.n_agents, axis=0)

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Cohort Intelligence over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        fitness = [agent.fit for agent in space.agents]

        for i, agent in enumerate(space.agents):
            s = g.weighted_wheel_selection(fitness)

            self.lower[i] = space.agents[s].position - self.lower[i] * self.r / 2
            self.upper[i] = space.agents[s].position - self.upper[i] * self.r / 2

            for _ in range(self.t):
                a = copy.deepcopy(agent)

                for j, (lb, ub) in enumerate(zip(self.lower[i], self.upper[i])):
                    a.position[j] = np.random.uniform(lb, ub, agent.n_dimensions)
                a.clip_by_bound()

                a.fit = function(a.position)
                if a.fit < agent.fit:
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

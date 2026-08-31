"""Differential Evolution."""

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class DE(Optimizer):
    """A DE class, inherited from Optimizer.

    This is the designed class to define DE-related
    variables and methods.

    References:
        R. Storn. On the usage of differential evolution for function optimization.
        Proceedings of North American Fuzzy Information Processing (1996).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(DE, self).__init__()

        self.CR = 0.9
        self.F = 0.7

        self.build(params)

    def _mutate_agent(
        self, agent: Agent, alpha: Agent, beta: Agent, gamma: Agent
    ) -> Agent:
        """Mutates a new agent based on pre-picked distinct agents (eq. 4).

        Args:
            agent: Current agent.
            alpha: 1st picked agent.
            beta: 2nd picked agent.
            gamma: 3rd picked agent.

        Returns:
            (Agent): A mutated agent.

        """

        a = copy.deepcopy(agent)

        R = np.random.randint(0, agent.n_variables, None)

        for j in range(a.n_variables):
            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 < self.CR or j == R:
                a.position[j] = alpha.position[j] + self.F * (
                    beta.position[j] - gamma.position[j]
                )

        return a

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Differential Evolution over all agents and variables (eq. 1-4).

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        for i, agent in enumerate(space.agents):
            C = np.random.choice(
                np.setdiff1d(range(0, len(space.agents)), i), 3, p=None, replace=False
            )

            a = self._mutate_agent(
                agent, space.agents[C[0]], space.agents[C[1]], space.agents[C[2]]
            )
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

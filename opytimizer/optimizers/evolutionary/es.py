"""Evolution Strategies."""

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class ES(Optimizer):
    """An ES class, inherited from Optimizer.

    This is the designed class to define ES-related
    variables and methods.

    References:
        T. Bäck and H.–P. Schwefel. An Overview of Evolutionary Algorithms for Parameter Optimization.
        Evolutionary Computation (1993).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(ES, self).__init__()

        self.child_ratio = 0.5

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.n_children = int(space.n_agents * self.child_ratio)
        self.strategy = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

        for i in range(self.n_children):
            for j, (lb, ub) in enumerate(zip(space.lb, space.ub)):
                self.strategy[i][j] = 0.05 * np.random.uniform(
                    0, ub - lb, space.agents[i].n_dimensions
                )

    def _mutate_parent(self, agent: Agent, index: int, function: Callable) -> Agent:
        """Mutates a parent into a new child (eq. 2).

        Args:
            agent: An agent instance to be reproduced.
            index: Index of current agent.
            function: A callable that will be used as the objective function.

        Returns:
            (Agent): A mutated child.

        """

        a = copy.deepcopy(agent)

        r1 = np.random.normal(0.0, 1.0, 1)
        a.position += self.strategy[index] * r1
        a.clip_by_bound()

        a.fit = function(a.position)

        return a

    def _update_strategy(self, index: int) -> np.ndarray:
        """Updates the strategy (eq. 5-10).

        Args:
            index: Index of current agent.

        Returns:
            (np.ndarray): The updated strategy.

        """

        n_variables, n_dimensions = self.strategy.shape[1], self.strategy.shape[2]

        tau = 1 / np.sqrt(2 * n_variables)
        tau_p = 1 / np.sqrt(2 * np.sqrt(n_variables))

        r1 = np.random.normal(0.0, 1.0, (n_variables, n_dimensions))
        r2 = np.random.normal(0.0, 1.0, (n_variables, n_dimensions))

        self.strategy[index] *= np.exp(tau_p * r1 + tau * r2)

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Evolution Strategies over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        n_agents = len(space.agents)

        children = []
        for i in range(self.n_children):
            a = self._mutate_parent(space.agents[i], i, function)
            self._update_strategy(i)

            children.append(a)

        space.agents += children
        space.agents.sort(key=lambda x: x.fit)
        space.agents = space.agents[:n_agents]

"""Evolutionary Programming."""

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class EP(Optimizer):
    """An EP class, inherited from Optimizer.

    This is the designed class to define EP-related
    variables and methods.

    References:
        A. E. Eiben and J. E. Smith. Introduction to Evolutionary Computing.
        Natural Computing Series (2013).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(EP, self).__init__()

        self.bout_size = 0.1
        self.clip_ratio = 0.05

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.strategy = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

        for i in range(space.n_agents):
            for j, (lb, ub) in enumerate(zip(space.lb, space.ub)):
                self.strategy[i][j] = 0.05 * np.random.uniform(
                    0, ub - lb, space.agents[i].n_dimensions
                )

    def _mutate_parent(self, agent: Agent, index: int, function: Callable) -> Agent:
        """Mutates a parent into a new child (eq. 5.1).

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

    def _update_strategy(
        self, index: int, lower_bound: np.ndarray, upper_bound: np.ndarray
    ) -> np.ndarray:
        """Updates the strategy and performs a clipping process to help its convergence (eq. 5.2).

        Args:
            index: Index of current agent.
            lower_bound: An array holding the lower bounds.
            upper_bound: An array holding the upper bounds.

        Returns:
            (np.ndarray): The updated strategy.

        """

        n_variables, n_dimensions = self.strategy.shape[1], self.strategy.shape[2]

        r1 = np.random.normal(0.0, 1.0, (n_variables, n_dimensions))
        self.strategy[index] += r1 * (np.sqrt(np.abs(self.strategy[index])))

        for j, (lb, ub) in enumerate(zip(lower_bound, upper_bound)):
            self.strategy[index][j] = (
                np.clip(self.strategy[index][j], lb, ub) * self.clip_ratio
            )

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Evolutionary Programming over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        n_agents = len(space.agents)

        children = []
        for i, agent in enumerate(space.agents):
            a = self._mutate_parent(agent, i, function)
            self._update_strategy(i, agent.lb, agent.ub)

            children.append(a)

        space.agents += children

        n_individuals = int(n_agents * self.bout_size)
        wins = np.zeros(len(space.agents))

        for i, agent in enumerate(space.agents):
            for _ in range(n_individuals):
                index = np.random.randint(0, len(space.agents), None)
                if agent.fit < space.agents[index].fit:
                    wins[i] += 1

        space.agents = [
            agents
            for _, agents in sorted(
                zip(wins, space.agents), key=lambda pair: pair[0], reverse=True
            )
        ]
        space.agents = space.agents[:n_agents]

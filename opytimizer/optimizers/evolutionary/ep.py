"""Evolutionary Programming.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


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

        logger.info("Class overrided.")

    @property
    def bout_size(self) -> float:
        """Size of bout during the tournament selection."""

        return self._bout_size

    @bout_size.setter
    def bout_size(self, bout_size: float) -> None:
        if not isinstance(bout_size, (float, int)):
            raise e.TypeError("`bout_size` should be a float or integer")
        if bout_size < 0 or bout_size > 1:
            raise e.ValueError("`bout_size` should be between 0 and 1")

        self._bout_size = bout_size

    @property
    def clip_ratio(self) -> float:
        """Clipping ratio to helps the algorithm's convergence."""

        return self._clip_ratio

    @clip_ratio.setter
    def clip_ratio(self, clip_ratio: float) -> None:
        if not isinstance(clip_ratio, (float, int)):
            raise e.TypeError("`clip_ratio` should be a float or integer")
        if clip_ratio < 0 or clip_ratio > 1:
            raise e.ValueError("`clip_ratio` should be between 0 and 1")

        self._clip_ratio = clip_ratio

    @property
    def strategy(self) -> np.ndarray:
        """Array of strategies."""

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: np.ndarray) -> None:
        if not isinstance(strategy, np.ndarray):
            raise e.TypeError("`strategy` should be a numpy array")

        self._strategy = strategy

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
                self.strategy[i][j] = 0.05 * r.generate_uniform_random_number(
                    0, ub - lb, size=space.agents[i].n_dimensions
                )

    def _mutate_parent(self, agent: Agent, index: int, function: Function) -> Agent:
        """Mutates a parent into a new child (eq. 5.1).

        Args:
            agent: An agent instance to be reproduced.
            index: Index of current agent.
            function: A Function object that will be used as the objective function.

        Returns:
            (Agent): A mutated child.

        """

        a = copy.deepcopy(agent)

        r1 = r.generate_gaussian_random_number()

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

        r1 = r.generate_gaussian_random_number(size=(n_variables, n_dimensions))
        self.strategy[index] += r1 * (np.sqrt(np.abs(self.strategy[index])))

        for j, (lb, ub) in enumerate(zip(lower_bound, upper_bound)):
            self.strategy[index][j] = (
                np.clip(self.strategy[index][j], lb, ub) * self.clip_ratio
            )

    def update(self, space: Space, function: Function) -> None:
        """Wraps Evolutionary Programming over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

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
                index = r.generate_integer_random_number(0, len(space.agents))
                if agent.fit < space.agents[index].fit:
                    wins[i] += 1

        space.agents = [
            agents
            for _, agents in sorted(
                zip(wins, space.agents), key=lambda pair: pair[0], reverse=True
            )
        ]
        space.agents = space.agents[:n_agents]

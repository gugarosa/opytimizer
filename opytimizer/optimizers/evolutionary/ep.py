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

        # Overrides its parent class with the receiving params
        super(EP, self).__init__()

        # Size of bout during the tournament selection
        self.bout_size = 0.1

        # Clipping ratio to helps the algorithm's convergence
        self.clip_ratio = 0.05

        # Builds the class
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

        # Array of strategies
        self.strategy = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

        # Iterates through all agents
        for i in range(space.n_agents):
            # For every decision variable
            for j, (lb, ub) in enumerate(zip(space.lb, space.ub)):
                # Initializes the strategy array with the proposed EP distance
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

        # Makes a deep copy on selected agent
        a = copy.deepcopy(agent)

        # Generates a uniform random number
        r1 = r.generate_gaussian_random_number()

        # Updates its position
        a.position += self.strategy[index] * r1

        # Clips its limits
        a.clip_by_bound()

        # Calculates its fitness
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

        # Calculates the number of variables and dimensions
        n_variables, n_dimensions = self.strategy.shape[1], self.strategy.shape[2]

        # Generates a uniform random number
        r1 = r.generate_gaussian_random_number(size=(n_variables, n_dimensions))

        # Calculates the new strategy
        self.strategy[index] += r1 * (np.sqrt(np.abs(self.strategy[index])))

        # For every decision variable
        for j, (lb, ub) in enumerate(zip(lower_bound, upper_bound)):
            # Uses the clip ratio to help the convergence
            self.strategy[index][j] = (
                np.clip(self.strategy[index][j], lb, ub) * self.clip_ratio
            )

    def update(self, space: Space, function: Function) -> None:
        """Wraps Evolutionary Programming over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Calculates the number of agents
        n_agents = len(space.agents)

        # Creates a list for the produced children
        children = []

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Mutates a parent and generates a new child
            a = self._mutate_parent(agent, i, function)

            # Updates the strategy
            self._update_strategy(i, agent.lb, agent.ub)

            # Appends the mutated agent to the children
            children.append(a)

        # Joins both populations
        space.agents += children

        # Number of individuals to be selected
        n_individuals = int(n_agents * self.bout_size)

        # Creates an empty array of wins
        wins = np.zeros(len(space.agents))

        # Iterates through all agents in the new population
        for i, agent in enumerate(space.agents):
            # Iterate through all tournament individuals
            for _ in range(n_individuals):
                # Gathers a random index
                index = r.generate_integer_random_number(0, len(space.agents))

                # If current agent's fitness is smaller than selected one
                if agent.fit < space.agents[index].fit:
                    # Increases its winning by one
                    wins[i] += 1

        # Sorts agents list based on its winnings
        space.agents = [
            agents
            for _, agents in sorted(
                zip(wins, space.agents), key=lambda pair: pair[0], reverse=True
            )
        ]

        # Gathers the best `n_agents`
        space.agents = space.agents[:n_agents]

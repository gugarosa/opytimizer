"""Evolution Strategies.
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

        # Overrides its parent class with the receiving params
        super(ES, self).__init__()

        # Ratio of children in the population
        self.child_ratio = 0.5

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def child_ratio(self) -> float:
        """Ratio of children in the population."""

        return self._child_ratio

    @child_ratio.setter
    def child_ratio(self, child_ratio: float) -> None:
        if not isinstance(child_ratio, (float, int)):
            raise e.TypeError("`child_ratio` should be a float or integer")
        if child_ratio < 0 or child_ratio > 1:
            raise e.ValueError("`child_ratio` should be between 0 and 1")

        self._child_ratio = child_ratio

    @property
    def n_children(self) -> int:
        """Number of children."""

        return self._n_children

    @n_children.setter
    def n_children(self, n_children: int) -> None:
        if not isinstance(n_children, int):
            raise e.TypeError("`n_children` should be an integer")
        if n_children < 0:
            raise e.ValueError("`n_children` should be >= 0")

        self._n_children = n_children

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

        # Number of children and array of strategies
        self.n_children = int(space.n_agents * self.child_ratio)
        self.strategy = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

        # Iterates through all agents
        for i in range(self.n_children):
            # For every decision variable
            for j, (lb, ub) in enumerate(zip(space.lb, space.ub)):
                # Initializes the strategy array with the proposed EP distance
                self.strategy[i][j] = 0.05 * r.generate_uniform_random_number(
                    0, ub - lb, size=space.agents[i].n_dimensions
                )

    def _mutate_parent(self, agent: Agent, index: int, function: Function) -> Agent:
        """Mutates a parent into a new child (eq. 2).

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

    def _update_strategy(self, index: int) -> np.ndarray:
        """Updates the strategy (eq. 5-10).

        Args:
            index: Index of current agent.

        Returns:
            (np.ndarray): The updated strategy.

        """

        # Calculates the number of variables and dimensions
        n_variables, n_dimensions = self.strategy.shape[1], self.strategy.shape[2]

        # Calculates the mutation strength and its complementary
        tau = 1 / np.sqrt(2 * n_variables)
        tau_p = 1 / np.sqrt(2 * np.sqrt(n_variables))

        # Generates uniform random numbers
        r1 = r.generate_gaussian_random_number(size=(n_variables, n_dimensions))
        r2 = r.generate_gaussian_random_number(size=(n_variables, n_dimensions))

        # Calculates the new strategy
        self.strategy[index] *= np.exp(tau_p * r1 + tau * r2)

    def update(self, space: Space, function: Function) -> None:
        """Wraps Evolution Strategies over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Calculates the number of agents
        n_agents = len(space.agents)

        # Creates a list for the produced children
        children = []

        # Iterate through all children
        for i in range(self.n_children):
            # Mutates a parent and generates a new child
            a = self._mutate_parent(space.agents[i], i, function)

            # Updates the strategy
            self._update_strategy(i)

            # Appends the mutated agent to the children
            children.append(a)

        # Joins both populations, sorts agents and gathers best `n_agents`
        space.agents += children
        space.agents.sort(key=lambda x: x.fit)
        space.agents = space.agents[:n_agents]

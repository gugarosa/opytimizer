"""Crow Search Algorithm.
"""

import copy
import time
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class CSA(Optimizer):
    """A CSA class, inherited from Optimizer.

    This is the designed class to define CSA-related
    variables and methods.

    References:
        A. Askarzadeh. A novel metaheuristic method for
        solving constrained engineering optimization problems: Crow search algorithm.
        Computers & Structures (2016).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> CSA.")

        # Overrides its parent class with the receiving params
        super(CSA, self).__init__()

        # Flight length
        self.fl = 2.0

        # Awareness probability
        self.AP = 0.1

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def fl(self) -> float:
        """Flight length."""

        return self._fl

    @fl.setter
    def fl(self, fl: float) -> None:
        if not isinstance(fl, (float, int)):
            raise e.TypeError("`fl` should be a float or integer")

        self._fl = fl

    @property
    def AP(self) -> float:
        """Awareness probability."""

        return self._AP

    @AP.setter
    def AP(self, AP: float) -> None:
        if not isinstance(AP, (float, int)):
            raise e.TypeError("`AP` should be a float or integer")
        if AP < 0 or AP > 1:
            raise e.ValueError("`AP` should be between 0 and 1")

        self._AP = AP

    @property
    def memory(self) -> np.ndarray:
        """Array of memories."""

        return self._memory

    @memory.setter
    def memory(self, memory: np.ndarray) -> None:
        if not isinstance(memory, np.ndarray):
            raise e.TypeError("`memory` should be a numpy array")

        self._memory = memory

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Arrays of memories
        self.memory = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

    def evaluate(self, space: Space, function: Function) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A Space object that will be evaluated.
            function: A Function object that will be used as the objective function.

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Calculates the fitness value of current agent
            fit = function(agent.position)

            # If fitness is better than agent's best fit
            if fit < agent.fit:
                # Updates its current fitness to the newer one
                agent.fit = fit

                # Also updates the memory to current's agent position (eq. 5)
                self.memory[i] = copy.deepcopy(agent.position)

            # If agent's fitness is better than global fitness
            if agent.fit < space.best_agent.fit:
                # Makes a deep copy of agent's local best position and fitness to the best agent
                space.best_agent.position = copy.deepcopy(self.memory[i])
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: Space) -> None:
        """Wraps Crow Search Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        # Iterates through every agent
        for agent in space.agents:
            # Generates uniform random numbers
            r1 = r.generate_uniform_random_number()
            r2 = r.generate_uniform_random_number()

            # Generates a random integer (e.g. selects the crow)
            j = r.generate_integer_random_number(high=len(space.agents))

            # Checks if first random number is greater than awareness probability
            if r1 >= self.AP:
                # Updates agent's position (eq. 2)
                agent.position += r2 * self.fl * (self.memory[j] - agent.position)

            # If random number is smaller than probability
            else:
                # Fills agent with new random positions
                agent.fill_with_uniform()

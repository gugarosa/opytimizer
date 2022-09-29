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

        super(CSA, self).__init__()

        self.fl = 2.0
        self.AP = 0.1

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

        self.memory = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

    def evaluate(self, space: Space, function: Function) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A Space object that will be evaluated.
            function: A Function object that will be used as the objective function.

        """

        for i, agent in enumerate(space.agents):
            fit = function(agent.position)
            if fit < agent.fit:
                agent.fit = fit

                # Updates the memory to current's agent position (eq. 5)
                self.memory[i] = copy.deepcopy(agent.position)

            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(self.memory[i])
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: Space) -> None:
        """Wraps Crow Search Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        for agent in space.agents:
            r1 = r.generate_uniform_random_number()
            r2 = r.generate_uniform_random_number()

            # Generates a random integer (e.g. selects the crow)
            j = r.generate_integer_random_number(high=len(space.agents))

            if r1 >= self.AP:
                # Updates agent's position (eq. 2)
                agent.position += r2 * self.fl * (self.memory[j] - agent.position)
            else:
                agent.fill_with_uniform()

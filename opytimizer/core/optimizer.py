"""Optimizer.
"""

import copy
import time
from typing import Any, Dict

import opytimizer.utils.exception as e
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class Optimizer:
    """An Optimizer class that holds meta-heuristics-related properties
    and methods.

    """

    def __init__(self) -> None:
        """Initialization method."""

        self.algorithm = self.__class__.__name__
        self.params = {}

        self.built = False

    @property
    def algorithm(self) -> str:
        """str: Algorithm's name."""

        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm: str) -> None:
        if not isinstance(algorithm, str):
            raise e.TypeError("`algorithm` should be a string")

        self._algorithm = algorithm

    @property
    def built(self) -> bool:
        """Indicates whether the optimizer is built."""

        return self._built

    @built.setter
    def built(self, built: bool) -> None:
        if not isinstance(built, bool):
            raise e.TypeError("`built` should be a boolean")

        self._built = built

    @property
    def params(self) -> Dict[str, Any]:
        """Key-value parameters."""

        return self._params

    @params.setter
    def params(self, params: Dict[str, Any]) -> None:
        if not isinstance(params, dict):
            raise e.TypeError("`params` should be a dictionary")

        self._params = params

    def build(self, params: Dict[str, Any]) -> None:
        """Builds the object by creating its parameters.

        Args:
            params: Key-value parameters to the meta-heuristic.

        """

        if params:
            self.params = params

            for k, v in params.items():
                setattr(self, k, v)

        self.built = True

        logger.debug(
            "Algorithm: %s | Custom Parameters: %s | Built: %s.",
            self.algorithm,
            str(params),
            self.built,
        )

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        This method is called before the optimization procedure and makes sure
        that the additional variable is available as a property.

        """

        pass

    def evaluate(self, space: Space, function: Function) -> None:
        """Evaluates the search space according to the objective function.

        If you need a specific evaluate method, please re-implement
        it on child's class.

        Also, note that function only accept arguments that are
        found on Opytimizer class.

        Args:
            space: A Space object that will be evaluated.
            function: A Function object serving as an objective function.

        """

        for agent in space.agents:
            agent.fit = function(agent.position)

            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self) -> None:
        """Updates the agents' position array.

        As each child has a different procedure of update, you will need
        to implement it directly on its class.

        Also, note that function only accept arguments that are
        found on Opytimizer class.

        """

        pass

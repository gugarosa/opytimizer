"""Optimizer."""

import copy
import time
from typing import Any, Callable, Mapping, Optional

from opytimizer.core.space import Space


class Optimizer:
    """An Optimizer class that holds meta-heuristics-related properties
    and methods.

    """

    def build(self, params: Optional[Mapping[str, Any]] = None) -> None:
        """Builds the object by creating its parameters.

        Args:
            params: Key-value parameters to the meta-heuristic.

        """

        for key, value in (params or {}).items():
            setattr(self, key, value)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        This method is called before the optimization procedure and makes sure
        that the additional variable is available as a property.

        """

        pass

    def evaluate(self, space: Space, function: Callable) -> None:
        """Evaluates the search space according to the objective function.

        If you need a specific evaluate method, please re-implement
        it on child's class.

        Also, note that function only accept arguments that are
        found on Opytimizer class.

        Args:
            space: A Space object that will be evaluated.
            function: Objective callable.

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

"""Black Hole.
"""

from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.random as r
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import constant, logging

logger = logging.get_logger(__name__)


class BH(Optimizer):
    """A BH class, inherited from Optimizer.

    This is the designed class to define BH-related
    variables and methods.

    References:
        A. Hatamlou. Black hole: A new heuristic optimization approach for data clustering.
        Information Sciences (2013).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> BH.")

        super(BH, self).__init__()

        self.build(params)

        logger.info("Class overrided.")

    def _update_position(
        self, agents: List[Agent], best_agent: Agent, function: Function
    ) -> float:
        """It updates every star position and calculates their event's horizon cost (eq. 3).

        Args:
            agents: List of agents.
            best_agent: Global best agent.
            function: A function object.

        Returns:
            (float): The cost of the event horizon.

        """

        cost = 0

        for agent in agents:
            r1 = r.generate_uniform_random_number()
            agent.position += r1 * (best_agent.position - agent.position)
            agent.clip_by_bound()

            agent.fit = function(agent.position)
            if agent.fit < best_agent.fit:
                agent.position, best_agent.position = (
                    best_agent.position,
                    agent.position,
                )
                agent.fit, best_agent.fit = best_agent.fit, agent.fit

            cost += agent.fit

        return cost

    def _event_horizon(
        self, agents: List[Agent], best_agent: Agent, cost: float
    ) -> None:
        """It calculates the stars' crossing an event horizon (eq. 4).

        Args:
            agents: List of agents.
            best_agent: Global best agent.
            cost: The event's horizon cost.

        """

        radius = best_agent.fit / max(cost, constant.EPSILON)

        for agent in agents:
            distance = np.linalg.norm(best_agent.position - agent.position)
            if distance < radius:
                agent.fill_with_uniform()

    def update(self, space: Space, function: Function) -> None:
        """Wraps Black Hole over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Updates stars position and calculate their cost (eq. 3)
        cost = self._update_position(space.agents, space.best_agent, function)

        # Performs the Event Horizon (eq. 4)
        self._event_horizon(space.agents, space.best_agent, cost)

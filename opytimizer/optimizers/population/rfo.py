"""Red Fox Optimization."""

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class RFO(Optimizer):
    """A RFO class, inherited from Optimizer.

    This is the designed class to define RFO-related
    variables and methods.

    References:
        D. Polap and M. Woźniak. Red fox optimization algorithm.
        Expert Systems with Applications (2021).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(RFO, self).__init__()

        self.phi = np.random.uniform(0, 2 * np.pi, 1)[0]
        self.theta = np.random.uniform(0.0, 1.0, 1)[0]
        self.p_replacement = 0.05

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.n_replacement = int(self.p_replacement * space.n_agents)

    def _rellocation(self, agent: Agent, best_agent: Agent, function: Callable) -> None:
        """Performs the fox rellocation procedure.

        Args:
            agent: Current agent.
            best_agent: Best agent.
            function: A callable that will be used as the objective function.

        """

        temp = copy.deepcopy(agent)

        # Calculates the square root of euclidean distance between agent and best agent (eq. 1)
        distance = np.sqrt(np.linalg.norm(temp.position - best_agent.position))

        # Calculates individual reallocation (eq. 2)
        alpha = np.random.uniform(0, distance, 1)
        temp.position += alpha * np.sign(best_agent.position - temp.position)
        temp.clip_by_bound()

        temp.fit = function(temp.position)
        if temp.fit < agent.fit:
            agent.position = copy.deepcopy(temp.position)
            agent.fit = copy.deepcopy(temp.fit)

    def _noticing(self, agent: Agent, function: Callable, alpha: float) -> None:
        """Performs the fox noticing procedure.

        Args:
            agent: Current agent.
            function: A callable that will be used as the objective function.
            alpha: Scaling parameter.

        """

        mu = np.random.uniform(0.0, 1.0, 1)
        if mu > 0.75:
            if self.phi != 0:
                # Calculates fox observation radius (eq. 4 - top)
                radius = alpha * np.sin(self.phi) / self.phi
            else:
                # Calculates fox observation radius (eq. 4 - bottom)
                radius = self.theta

            phi = np.random.uniform(0, 2 * np.pi, agent.n_variables)

            for j in range(agent.n_variables):
                total_sum = 0

                for k in range(j):
                    total_sum += np.sin(phi[k])

                # Updates the corresponding position (eq. 5)
                agent.position[j] += alpha * radius * (total_sum + np.cos(phi[j]))
            agent.clip_by_bound()

            agent.fit = function(agent.position)

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Red Fox Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        alpha = np.random.uniform(0, 0.2, 1)

        for agent in space.agents:
            self._rellocation(agent, space.best_agent, function)
            self._noticing(agent, function, alpha)

        space.agents.sort(key=lambda x: x.fit)

        # Calculates the habitat's center and diameter (eq. 6 and 7)
        habitat_center = (space.agents[0].position + space.agents[1].position) / 2
        habitat_diameter = np.sqrt(
            np.linalg.norm(space.agents[0].position - space.agents[1].position)
        )

        k = np.random.uniform(0.0, 1.0, 1)

        for agent in space.agents[-self.n_replacement :]:
            # If sampled number is bigger than 0.45 (eq. 8 - top)
            if k >= 0.45:
                agent.fill_with_uniform()
                agent.position += habitat_center + habitat_diameter / 2

            # If sampled number is smaller than 0.45 (eq. 8 - bottom)
            else:
                # Reproduces parents into a new position (eq. 9)
                agent.position = (
                    k * (space.agents[0].position + space.agents[1].position) / 2
                )

            agent.clip_by_bound()

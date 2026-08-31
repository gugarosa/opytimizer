"""Pigeon-Inspired Optimization."""

from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class PIO(Optimizer):
    """A PIO class, inherited from Optimizer.

    This is the designed class to define PIO-related
    variables and methods.

    References:
        H. Duan and P. Qiao.
        Pigeon-inspired optimization:a new swarm intelligence optimizerfor air robot path planning.
        International Journal of IntelligentComputing and Cybernetics (2014).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(PIO, self).__init__()

        self.n_c1 = 150
        self.n_c2 = 200

        self.R = 0.2

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.n_p = space.n_agents

        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def _calculate_center(self, agents: List[Agent]) -> np.ndarray:
        """Calculates the center position (eq. 8).

        Args:
            agents: List of agents.

        Returns:
            (np.ndarray): The center position.

        """

        total_pos = np.zeros((agents[0].n_variables, agents[0].n_dimensions))
        total_fit = 0.0

        for agent in agents:
            total_pos += agent.position * agent.fit
            total_fit += agent.fit

        center = total_pos / (self.n_p * total_fit + c.EPSILON)

        return center

    def _update_center_position(self, position: np.ndarray, center: np.ndarray) -> None:
        """Updates a pigeon position based on the center (eq. 9).

        Args:
            position: Agent's current position.
            center: Center position.

        Returns:
            (np.ndarray): A new center-based position.

        """

        r1 = np.random.uniform(0.0, 1.0, 1)
        new_position = position + r1 * (center - position)

        return new_position

    def update(self, space: Space, iteration: int) -> None:
        """Wraps Pigeon-Inspired Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.

        """

        if iteration < self.n_c1:
            for i, agent in enumerate(space.agents):
                # Updates current agent velocity (eq. 5)
                r1 = np.random.uniform(0.0, 1.0, 1)
                self.velocity[i] = self.velocity[i] * np.exp(
                    -self.R * (iteration + 1)
                ) + r1 * (space.best_agent.position - agent.position)

                # Updates current agent position (eq. 6)
                agent.position += self.velocity[i]
        elif iteration < self.n_c2:
            # Calculates the number of possible pigeons (eq. 7)
            self.n_p = int(self.n_p / 2) + 1

            space.agents.sort(key=lambda x: x.fit)
            center = self._calculate_center(space.agents[: self.n_p])

            for agent in space.agents:
                agent.position = self._update_center_position(agent.position, center)

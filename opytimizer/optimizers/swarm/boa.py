"""Butterfly Optimization Algorithm."""

from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class BOA(Optimizer):
    """A BOA class, inherited from Optimizer.

    This is the designed class to define BOA-related
    variables and methods.

    References:
        S. Arora and S. Singh. Butterfly optimization algorithm: a novel approach for global optimization.
        Soft Computing (2019).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(BOA, self).__init__()

        self.c = 0.01
        self.a = 0.1
        self.p = 0.8

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.fragrance = np.zeros(space.n_agents)

    def _best_movement(
        self,
        agent_position: np.ndarray,
        best_position: np.ndarray,
        fragrance: np.ndarray,
        random: float,
    ) -> np.ndarray:
        """Updates the agent's position towards the best butterfly (eq. 2).

        Args:
            agent_positio: Agent's current position.
            best_positio: Best agent's current position.
            fragrance: Agent's current fragrance value.
            random: A random number between 0 and 1.

        Returns:
            (np.ndarray): A new position based on best movement.

        """

        new_position = (
            agent_position + (random**2 * best_position - agent_position) * fragrance
        )

        return new_position

    def _local_movement(
        self,
        agent_position: np.ndarray,
        j_position: np.ndarray,
        k_position: np.ndarray,
        fragrance: np.ndarray,
        random: float,
    ) -> np.ndarray:
        """Updates the agent's position using a local movement (eq. 3).

        Args:
            agent_positio: Agent's current position.
            j_positio: Agent `j` current position.
            k_positio: Agent `k` current position.
            fragrance: Agent's current fragrance value.
            random: A random number between 0 and 1.

        Returns:
            (np.ndarray): A new position based on local movement.

        """

        new_position = (
            agent_position + (random**2 * j_position - k_position) * fragrance
        )

        return new_position

    def update(self, space: Space) -> None:
        """Wraps Butterfly Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        for i, agent in enumerate(space.agents):
            # Calculates fragrance for current agent (eq. 1)
            self.fragrance[i] = self.c * agent.fit**self.a

        for i, agent in enumerate(space.agents):
            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 < self.p:
                # Moves current agent towards the best one (eq. 2)
                agent.position = self._best_movement(
                    agent.position, space.best_agent.position, self.fragrance[i], r1
                )
            else:
                j = np.random.randint(0, len(space.agents), None)
                k = r.integer(0, len(space.agents), exclude=j, size=None)

                # Moves current agent using a local movement (eq. 3)
                agent.position = self._local_movement(
                    agent.position,
                    space.agents[j].position,
                    space.agents[k].position,
                    self.fragrance[i],
                    r1,
                )

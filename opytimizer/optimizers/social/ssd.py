"""Social Ski Driver."""

import copy
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class SSD(Optimizer):
    """An SSD class, inherited from Optimizer.

    This is the designed class to define SSD-related
    variables and methods.

    References:
        A. Tharwat and T. Gabel.
        Parameters optimization of support vector machines for imbalanced data using social ski driver algorithm.
        Neural Computing and Applications (2019).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(SSD, self).__init__()

        self.c = 2.0
        self.decay = 0.99

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )
        self.velocity = np.random.uniform(
            0.0, 1.0, (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def _mean_global_solution(
        self, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray
    ) -> np.ndarray:
        """Calculates the mean global solution (eq. 9).

        Args:
            alpha: 1st agent's current position.
            beta: 2nd agent's current position.
            gamma: 3rd agent's current position.

        Returns:
            (np.ndarray): Mean global solution.

        """

        mean = (alpha + beta + gamma) / 3

        return mean

    def _update_position(self, position: np.ndarray, index: int) -> np.ndarray:
        """Updates a particle position (eq. 10).

        Args:
            position: Agent's current position.
            index: Index of current agent.

        Returns:
            (np.ndarray): A new position.

        """

        new_position = position + self.velocity[index]

        return new_position

    def _update_velocity(
        self, position: np.ndarray, mean: np.ndarray, index: int
    ) -> np.ndarray:
        """Updates a particle velocity (eq. 11).

        Args:
            position: Agent's current position.
            mean: Mean global best position.
            index: Index of current agent.

        Returns:
            (np.ndarray): A new velocity.

        """

        r1 = np.random.uniform(0.0, 1.0, 1)
        r2 = np.random.uniform(0.0, 1.0, 1)

        if r2 <= 0.5:
            new_velocity = self.c * np.sin(r1) * (
                self.local_position[index] - position
            ) + np.sin(r1) * (mean - position)
        else:
            new_velocity = self.c * np.cos(r1) * (
                self.local_position[index] - position
            ) + np.cos(r1) * (mean - position)

        return new_velocity

    def evaluate(self, space: Space, function: Callable) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A Space object that will be evaluated.
            function: A callable that will be used as the objective function.

        """

        for i, agent in enumerate(space.agents):
            fit = function(agent.position)
            if fit < agent.fit:
                agent.fit = fit

                self.local_position[i] = copy.deepcopy(agent.position)

            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(self.local_position[i])
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Social Ski Driver over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        for i, agent in enumerate(space.agents):
            fit = function(agent.position)
            if fit < agent.fit:
                agent.fit = fit

                self.local_position[i] = copy.deepcopy(agent.position)

            space.agents.sort(key=lambda x: x.fit)

            mean = self._mean_global_solution(
                space.agents[0].position,
                space.agents[1].position,
                space.agents[2].position,
            )

            agent.position = self._update_position(agent.position, i)
            agent.clip_by_bound()

            self.velocity[i] = self._update_velocity(agent.position, mean, i)

        self.c *= self.decay

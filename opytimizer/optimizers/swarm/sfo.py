"""Sailfish Optimizer."""

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class SFO(Optimizer):
    """A SFO class, inherited from Optimizer.

    This is the designed class to define SFO-related
    variables and methods.

    References:
        S. Shadravan, H. Naji and V. Bardsiri.
        The Sailfish Optimizer: A novel nature-inspired metaheuristic algorithm
        for solving constrained engineering optimization problems.
        Engineering Applications of Artificial Intelligence (2019).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(SFO, self).__init__()

        self.PP = 0.1
        self.A = 4
        self.e = 0.001

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.sardines = [
            self._generate_random_agent(space.best_agent)
            for _ in range(int(space.n_agents / self.PP))
        ]
        self.sardines.sort(key=lambda x: x.fit)

    def _generate_random_agent(self, agent: Agent) -> Agent:
        """Generates a new random-based agent.

        Args:
            agent: Agent to be copied.

        Returns:
            (Agent): Random-based agent.

        """

        a = copy.deepcopy(agent)
        a.fill_with_uniform()

        return a

    def _calculate_lambda_i(self, n_sailfishes: int, n_sardines: int) -> float:
        """Calculates the lambda value (eq. 7).

        Args:
            n_sailfishes (int): Number of sailfishes.
            n_sardines (int): Number of sardines.

        Returns:
            (float): Lambda value from current iteration.

        """

        # Calculates the prey density (eq. 8)
        PD = 1 - (n_sailfishes / (n_sailfishes + n_sardines))

        r1 = np.random.uniform(0.0, 1.0, 1)
        lambda_i = 2 * r1 * PD - PD

        return lambda_i

    def _update_sailfish(
        self, agent: Agent, best_agent: Agent, best_sardine: Agent, lambda_i: float
    ) -> np.ndarray:
        """Updates the sailfish's position (eq. 6).

        Args:
            agent: Current agent's.
            best_agent: Best sailfish.
            best_sardine: Best sardine.
            lambda_i: Lambda value.

        Returns:
            (np.ndarray): An updated position.

        """

        r1 = np.random.uniform(0.0, 1.0, 1)
        new_position = best_sardine.position - lambda_i * (
            r1 * (best_agent.position - best_sardine.position) / 2 - agent.position
        )

        return new_position

    def update(self, space: Space, function: Callable, iteration: int) -> None:
        """Wraps Sailfish Optimizer over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.
            iteration: Current iteration.

        """

        best_sardine = self.sardines[0]

        n_sailfishes = len(space.agents)
        n_sardines = len(self.sardines)
        n_variables = space.agents[0].n_variables

        for agent in space.agents:
            lambda_i = self._calculate_lambda_i(n_sailfishes, n_sardines)

            agent.position = self._update_sailfish(
                agent, space.best_agent, best_sardine, lambda_i
            )
            agent.clip_by_bound()

            agent.fit = function(agent.position)

        # Calculates the attack power (eq. 10)
        AP = np.fabs(self.A * (1 - 2 * iteration * self.e))

        if AP < 0.5:
            # Calculates the number of sardines possible replacements (eq. 11)
            alpha = int(len(self.sardines) * AP)

            # Calculates the number of variables possible replacements (eq. 12)
            beta = int(n_variables * AP)

            selected_sardines = np.random.randint(0, n_sardines, alpha)

            for i in selected_sardines:
                selected_vars = np.random.randint(0, n_variables, beta)

                for j in selected_vars:
                    r1 = np.random.uniform(0.0, 1.0, 1)

                    # Updates the sardine's position (eq. 9)
                    self.sardines[i].position[j] = r1 * (
                        space.best_agent.position[j] - self.sardines[i].position[j] + AP
                    )
                self.sardines[i].clip_by_bound()

                self.sardines[i].fit = function(self.sardines[i].position)
        else:
            for sardine in self.sardines:
                # Updates the sardine's position (eq. 9)
                r1 = np.random.uniform(0.0, 1.0, 1)
                sardine.position = r1 * (
                    space.best_agent.position - sardine.position + AP
                )
                sardine.clip_by_bound()

                # Re-calculates its fitness
                sardine.fit = function(sardine.position)

        space.agents.sort(key=lambda x: x.fit)
        self.sardines.sort(key=lambda x: x.fit)

        for agent in space.agents:
            for sardine in self.sardines:
                # If agent is worse than sardine (eq. 13)
                if agent.fit > sardine.fit:
                    agent = copy.deepcopy(sardine)
                    break

"""Equilibrium Optimizer."""

import copy
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class EO(Optimizer):
    """An EO class, inherited from Optimizer.

    This is the designed class to define EO-related
    variables and methods.

    References:
        A. Faramarzi et al. Equilibrium optimizer: A novel optimization algorithm.
        Knowledge-Based Systems (2020).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(EO, self).__init__()

        self.a1 = 2.0
        self.a2 = 1.0
        self.GP = 0.5
        self.V = 1.0

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.C = [copy.deepcopy(space.agents[0]) for _ in range(4)]

    def _calculate_equilibrium(self, agents: List[Agent]) -> None:
        """Calculates the equilibrium concentrations.

        Args:
            agents: List of agents.

        """

        for agent in agents:
            if agent.fit < self.C[0].fit:
                self.C[0] = copy.deepcopy(agent)
            elif agent.fit < self.C[1].fit:
                self.C[1] = copy.deepcopy(agent)
            elif agent.fit < self.C[2].fit:
                self.C[2] = copy.deepcopy(agent)
            elif agent.fit < self.C[3].fit:
                self.C[3] = copy.deepcopy(agent)

    def _average_concentration(self, function: Callable) -> Agent:
        """Averages the concentrations.

        Args:
            function: A callable that will be used as the objective function.

        Returns:
            (Agent): Averaged concentration.

        """

        C_avg = copy.deepcopy(self.C[0])
        C_avg.position = np.mean([c.position for c in self.C], axis=0)
        C_avg.clip_by_bound()

        C_avg.fit = function(C_avg.position)

        return C_avg

    def update(
        self, space: Space, function: Callable, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Equilibrium Optimizer over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        self._calculate_equilibrium(space.agents)
        C_avg = self._average_concentration(function)

        # Makes a pool of both concentrations and their average (eq. 7)
        C_pool = self.C + [C_avg]

        # Calculates the time (eq. 9)
        t = (1 - iteration / n_iterations) ** (self.a2 * iteration / n_iterations)

        for agent in space.agents:
            i = np.random.randint(0, 5, None)

            # Generates two uniform random vectors (eq. 11)
            r = np.random.uniform(0.0, 1.0, (agent.n_variables, agent.n_dimensions))
            lambd = np.random.uniform(0.0, 1.0, (agent.n_variables, agent.n_dimensions))

            # Calculates the exponential term (eq. 11)
            F = self.a1 * np.sign(r - 0.5) * (np.exp(-lambd * t) - 1)

            r1 = np.random.uniform(0.0, 1.0, 1)
            r2 = np.random.uniform(0.0, 1.0, 1)

            # If `r2` is bigger than generation probability (eq. 15)
            if r2 >= self.GP:
                GCP = 0.5 * r1
            else:
                GCP = 0

            # Calculates the initial generation value (eq. 14)
            G_0 = GCP * (C_pool[i].position - lambd * agent.position)

            # Calculates the generation value (eq. 13)
            G = G_0 * F

            # Updates agent's position (eq. 16)
            agent.position = (
                C_pool[i].position
                + (agent.position - C_pool[i].position) * F
                + (G / (lambd * self.V)) * (1 - F)
            )

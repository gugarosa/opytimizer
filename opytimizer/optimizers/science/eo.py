"""Equilibrium Optimizer.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.random as rnd
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


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

        logger.info("Overriding class: Optimizer -> EO.")

        super(EO, self).__init__()

        self.a1 = 2.0
        self.a2 = 1.0
        self.GP = 0.5
        self.V = 1.0

        self.build(params)

        logger.info("Class overrided.")

    @property
    def a1(self) -> float:
        """Exploration constant."""

        return self._a1

    @a1.setter
    def a1(self, a1: float) -> None:
        if not isinstance(a1, (float, int)):
            raise e.TypeError("`a1` should be a float or integer")
        if a1 < 0:
            raise e.ValueError("`a1` should be >= 0")

        self._a1 = a1

    @property
    def a2(self) -> float:
        """Exploitation constant."""

        return self._a2

    @a2.setter
    def a2(self, a2: float) -> None:
        if not isinstance(a2, (float, int)):
            raise e.TypeError("`a2` should be a float or integer")
        if a2 < 0:
            raise e.ValueError("`a2` should be >= 0")

        self._a2 = a2

    @property
    def GP(self) -> float:
        """Generation probability."""

        return self._GP

    @GP.setter
    def GP(self, GP: float) -> None:
        if not isinstance(GP, (float, int)):
            raise e.TypeError("`GP` should be a float or integer")
        if GP < 0 or GP > 1:
            raise e.ValueError("`GP` should be between 0 and 1")

        self._GP = GP

    @property
    def V(self) -> float:
        """Velocity."""

        return self._V

    @V.setter
    def V(self, V: float) -> None:
        if not isinstance(V, (float, int)):
            raise e.TypeError("`V` should be a float or integer")
        if V < 0:
            raise e.ValueError("`V` should be >= 0")

        self._V = V

    @property
    def C(self) -> List[Agent]:
        """Concentrations (agents)."""

        return self._C

    @C.setter
    def C(self, C: List[Agent]) -> None:
        if not isinstance(C, list):
            raise e.TypeError("`C` should be a list")

        self._C = C

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

    def _average_concentration(self, function: Function) -> Agent:
        """Averages the concentrations.

        Args:
            function: A Function object that will be used as the objective function.

        Returns:
            (Agent): Averaged concentration.

        """

        C_avg = copy.deepcopy(self.C[0])
        C_avg.position = np.mean([c.position for c in self.C], axis=0)
        C_avg.clip_by_bound()

        C_avg.fit = function(C_avg.position)

        return C_avg

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Equilibrium Optimizer over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
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
            i = rnd.generate_integer_random_number(0, 5)

            # Generates two uniform random vectors (eq. 11)
            r = rnd.generate_uniform_random_number(
                size=(agent.n_variables, agent.n_dimensions)
            )
            lambd = rnd.generate_uniform_random_number(
                size=(agent.n_variables, agent.n_dimensions)
            )

            # Calculates the exponential term (eq. 11)
            F = self.a1 * np.sign(r - 0.5) * (np.exp(-lambd * t) - 1)

            r1 = rnd.generate_uniform_random_number()
            r2 = rnd.generate_uniform_random_number()

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

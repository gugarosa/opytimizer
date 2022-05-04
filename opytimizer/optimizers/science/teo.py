"""Thermal Exchange Optimization.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class TEO(Optimizer):
    """A TEO class, inherited from Optimizer.

    This is the designed class to define TEO-related
    variables and methods.

    References:
        A. Kaveh and A. Dadras. A novel meta-heuristic optimization algorithm: Thermal exchange optimization.
        Advances in Engineering Software (2017).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> TEO.")

        # Overrides its parent class with the receiving params
        super(TEO, self).__init__()

        # Random step size control
        self.c1 = True

        # Randomness control
        self.c2 = True

        # Cooling parameter
        self.pro = 0.05

        # Thermal memory size
        self.n_TM = 4

        # Thermal memory
        self.TM = []

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def c1(self) -> bool:
        """Random step size control."""

        return self._c1

    @c1.setter
    def c1(self, c1: bool) -> None:
        if not isinstance(c1, bool):
            raise e.TypeError("`c1` should be a bool")

        self._c1 = c1

    @property
    def c2(self) -> bool:
        """Randomness control."""

        return self._c2

    @c2.setter
    def c2(self, c2: bool) -> None:
        if not isinstance(c2, bool):
            raise e.TypeError("`c2` should be a bool")

        self._c2 = c2

    @property
    def pro(self) -> float:
        """Cooling parameter."""

        return self._pro

    @pro.setter
    def pro(self, pro: float) -> None:
        if not isinstance(pro, (float, int)):
            raise e.TypeError("`pro` should be a float or integer")
        if pro < 0 or pro > 1:
            raise e.ValueError("`pro` should be between 0 and 1")

        self._pro = pro

    @property
    def n_TM(self) -> int:
        """Size of thermal memory."""

        return self._n_TM

    @n_TM.setter
    def n_TM(self, n_TM: int) -> None:
        if not isinstance(n_TM, int):
            raise e.TypeError("`n_TM` should be an integer")
        if n_TM <= 0:
            raise e.ValueError("`n_TM` should be > 0")

        self._n_TM = n_TM

    @property
    def TM(self) -> List[Agent]:
        """Thermal memory."""

        return self._TM

    @TM.setter
    def TM(self, TM: List[Agent]) -> None:
        if not isinstance(TM, list):
            raise e.TypeError("`TM` should be a list")

        self._TM = TM

    @property
    def environment(self) -> List[Agent]:
        """Environmental population."""

        return self._environment

    @environment.setter
    def environment(self, environment: List[Agent]) -> None:
        if not isinstance(environment, list):
            raise e.TypeError("`environment` should be a list")

        self._environment = environment

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Creates an enviromental population as a copy of initial population
        self.environment = copy.deepcopy(space.agents)

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        """Wraps Thermal Exchange Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Sorts agents
        space.agents.sort(key=lambda x: x.fit)

        # Updates the thermal memory and cuts it to maximum allowed size
        self.TM.append(copy.deepcopy(space.agents[0]))
        self.TM = self.TM[-self.n_TM :]

        # Replaces the worst agents with the thermal memory and re-sorts the agents
        space.agents = space.agents[: -len(self.TM)] + self.TM
        space.agents.sort(key=lambda x: x.fit)

        # Calculates the time (eq. 9)
        time = iteration / n_iterations

        # Iterates through all environmental-based agents
        for env in self.environment:
            # Updates the environment's position (eq. 10)
            r1 = r.generate_uniform_random_number()
            env.position = 1 - (self.c1 + self.c2 * (1 - time)) * r1 * env.position

        # Iterates through both populations' agents
        for agent, env in zip(space.agents, self.environment):
            # Calculates the agent's beta value (eq. 8)
            beta = agent.fit / space.agents[-1].fit

            # Updates the agent's position (eq. 11)
            agent.position = env.position + (agent.position - env.position) * np.exp(
                -beta * time
            )

            # Generates a random number
            r1 = r.generate_uniform_random_number()

            # If random number is smaller than `pro`
            if r1 < self.pro:
                # Selects a random dimension
                idx = r.generate_integer_random_number(high=agent.n_variables)

                # Resets its position (eq. 12)
                r2 = r.generate_uniform_random_number()
                agent.position[idx] = agent.lb[idx] + r2 * (
                    agent.ub[idx] - agent.lb[idx]
                )

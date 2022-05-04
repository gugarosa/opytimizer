"""Artificial Flora.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class AF(Optimizer):
    """An AF class, inherited from Optimizer.

    This is the designed class to define AF-related
    variables and methods.

    References:
        L. Cheng, W. Xue-han and Y. Wang. Artificial flora (AF) optimization algorithm.
        Applied Sciences (2018).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> AF.")

        # Overrides its parent class with the receiving params
        super(AF, self).__init__()

        # First learning coefficient
        self.c1 = 0.75

        # Second learning coefficient
        self.c2 = 1.25

        # Amount of branches
        self.m = 10

        # Selective probability
        self.Q = 0.75

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def c1(self) -> float:
        """First learning coefficient."""

        return self._c1

    @c1.setter
    def c1(self, c1: float) -> None:
        if not isinstance(c1, (float, int)):
            raise e.TypeError("`c1` should be a float or integer")
        if c1 < 0:
            raise e.ValueError("`c1` should be >= 0")

        self._c1 = c1

    @property
    def c2(self) -> float:
        """Second learning coefficient."""

        return self._c2

    @c2.setter
    def c2(self, c2: float) -> None:
        if not isinstance(c2, (float, int)):
            raise e.TypeError("`c2` should be a float or integer")
        if c2 < 0:
            raise e.ValueError("`c2` should be >= 0")

        self._c2 = c2

    @property
    def m(self) -> int:
        """Amount of branches."""

        return self._m

    @m.setter
    def m(self, m: int) -> None:
        if not isinstance(m, int):
            raise e.TypeError("`m` should be an integer")
        if m <= 0:
            raise e.ValueError("`m` should be > 0")

        self._m = m

    @property
    def Q(self) -> float:
        """Selective probability."""

        return self._Q

    @Q.setter
    def Q(self, Q: float) -> None:
        if not isinstance(Q, (float, int)):
            raise e.TypeError("`Q` should be a float or integer")
        if Q < 0 or Q > 1:
            raise e.ValueError("`Q` should be between 0 and 1")

        self._Q = Q

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Array of parent distances
        self.p_distance = r.generate_uniform_random_number(size=space.n_agents)

        # Array of grand-parent distances
        self.g_distance = r.generate_uniform_random_number(size=space.n_agents)

    def update(self, space: Space, function: Function) -> None:
        """Wraps Artificial Flora over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Sorts the agents
        space.agents.sort(key=lambda x: x.fit)

        # Creates a list of new agents
        new_agents = []

        # Iterates thorugh all agents
        for i, agent in enumerate(space.agents):
            # Iterates through amount of branches
            for _ in range(self.m):
                # Makes a copy of current agent
                a = copy.deepcopy(agent)

                # Generates random numbers
                r1 = r.generate_uniform_random_number()
                r2 = r.generate_uniform_random_number()
                r3 = r.generate_uniform_random_number()

                # Calculates the new distance (eq. 1)
                distance = (
                    self.g_distance[i] * r1 * self.c1
                    + self.p_distance[i] * r2 * self.c2
                )

                # Generates a random gaussian number
                D = r.generate_gaussian_random_number(
                    0, distance, (space.n_variables, space.n_dimensions)
                )

                # Updates offspring's position (eq. 5)
                a.position += D

                # Clips its limits
                a.clip_by_bound()

                # Evaluates its fitness
                a.fit = function(a.position)

                # Calculates the probability of selection (eq. 6)
                p = np.fabs(np.sqrt(a.fit / space.agents[-1].fit)) * self.Q

                # If random number is smaller than probability of selection
                if r3 < p:
                    # Appends the offsprings
                    new_agents.append(a)

            # Updates both grandparent and parent distances (eq. 2 and 3)
            self.g_distance[i] = self.p_distance[i]
            self.p_distance[i] = np.std(agent.position - a.position)

        # Randomly selects the agents
        idx = d.generate_choice_distribution(len(new_agents), None, space.n_agents)
        space.agents = [new_agents[i] for i in idx]

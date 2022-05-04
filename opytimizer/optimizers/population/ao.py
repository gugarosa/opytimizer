"""Aquila Optimizer.
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


class AO(Optimizer):
    """An AO class, inherited from Optimizer.

    This is the designed class to define AO-related
    variables and methods.

    References:
        L. Abualigah et al. Aquila Optimizer: A novel meta-heuristic optimization Algorithm.
        Computers & Industrial Engineering (2021).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> AO.")

        # Overrides its parent class with the receiving params
        super(AO, self).__init__()

        # First exploitation adjustment coefficient
        self.alpha = 0.1

        # Second exploitation adjustment coefficient
        self.delta = 0.1

        # Number of search cycles
        self.n_cycles = 10

        # Cycle regularizer
        self.U = 0.00565

        # Angle regularizer
        self.w = 0.005

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        """First exploitation adjustment coefficient."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        if alpha < 0:
            raise e.ValueError("`alpha` should be >= 0")

        self._alpha = alpha

    @property
    def delta(self) -> float:
        """Second exploitation adjustment coefficient."""

        return self._delta

    @delta.setter
    def delta(self, delta: float) -> None:
        if not isinstance(delta, (float, int)):
            raise e.TypeError("`delta` should be a float or integer")
        if delta < 0:
            raise e.ValueError("`delta` should be >= 0")

        self._delta = delta

    @property
    def n_cycles(self) -> int:
        """Number of cycles."""

        return self._n_cycles

    @n_cycles.setter
    def n_cycles(self, n_cycles: int) -> None:
        if not isinstance(n_cycles, int):
            raise e.TypeError("`n_cycles` should be an integer")
        if n_cycles <= 0:
            raise e.ValueError("`n_cycles` should be > 0")

        self._n_cycles = n_cycles

    @property
    def U(self) -> float:
        """Cycle regularizer."""

        return self._U

    @U.setter
    def U(self, U: float) -> None:
        if not isinstance(U, (float, int)):
            raise e.TypeError("`U` should be a float or integer")
        if U < 0:
            raise e.ValueError("`U` should be >= 0")

        self._U = U

    @property
    def w(self) -> float:
        """Angle regularizer."""

        return self._w

    @w.setter
    def w(self, w: float) -> None:
        if not isinstance(w, (float, int)):
            raise e.TypeError("`w` should be a float or integer")
        if w < 0:
            raise e.ValueError("`w` should be >= 0")

        self._w = w

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Aquila Optimizer over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Calculates the mean position of space
        average = np.mean([agent.position for agent in space.agents], axis=0)

        # Iterates through all agents
        for agent in space.agents:
            # Makes a deep copy of current agent
            a = copy.deepcopy(agent)

            # Generates a random number
            r1 = r.generate_uniform_random_number()

            # If current iteration is smaller than 2/3 of maximum iterations
            if iteration <= ((2 / 3) * n_iterations):
                # Generates another random number
                r2 = r.generate_uniform_random_number()

                # If random number is smaller or equal to 0.5
                if r1 <= 0.5:
                    # Updates temporary agent's position (eq. 3)
                    a.position = space.best_agent.position * (
                        1 - (iteration / n_iterations)
                    ) + (average - space.best_agent.position * r2)

                # If random number is bigger than 0.5
                else:
                    # Generates a Lévy distirbution and a random integer
                    levy = d.generate_levy_distribution(
                        size=(agent.n_variables, agent.n_dimensions)
                    )
                    idx = r.generate_integer_random_number(high=len(space.agents))

                    # Creates an evenly-space array of `n_variables`
                    # Also broadcasts it to correct `n_dimensions` size
                    D = np.linspace(1, agent.n_variables, agent.n_variables)
                    D = np.repeat(np.expand_dims(D, -1), agent.n_dimensions, axis=1)

                    # Calculates current cycle value (eq. 10)
                    cycle = self.n_cycles + self.U * D

                    # Calculates `theta` (eq. 11)
                    theta = -self.w * D + (3 * np.pi) / 2

                    # Calculates `x` and `y` positioning (eq. 8 and 9)
                    x = cycle * np.sin(theta)
                    y = cycle * np.cos(theta)

                    # Updates temporary agent's position (eq. 5)
                    a.position = (
                        space.best_agent.position * levy
                        + space.agents[idx].position
                        + (y - x) * r2
                    )

            # If current iteration is bigger than 2/3 of maximum iterations
            else:
                # Generates another random number
                r2 = r.generate_uniform_random_number()

                # If random number is smaller or equal to 0.5
                if r1 <= 0.5:
                    # Expands both lower and upper bound dimensions
                    lb = np.expand_dims(agent.lb, -1)
                    ub = np.expand_dims(agent.ub, -1)

                    # Updates temporary agent's position (eq. 13)
                    a.position = (
                        (space.best_agent.position - average) * self.alpha
                        - r2
                        + ((ub - lb) * r2 + lb) * self.delta
                    )

                # If random number is bigger than 0.5
                else:
                    # Calculates both motions (eq. 16 and 17)
                    G1 = 2 * r2 - 1
                    G2 = 2 * (1 - (iteration / n_iterations))

                    # Calculates quality function (eq. 15)
                    QF = iteration ** (G1 / (1 - n_iterations) ** 2)

                    # Generates a Lévy distribution
                    levy = d.generate_levy_distribution(
                        size=(agent.n_variables, agent.n_dimensions)
                    )

                    # Updates temporary agent's position (eq. 14)
                    a.position = (
                        QF * space.best_agent.position
                        - (G1 * a.position * r2)
                        - G2 * levy
                        + r2 * G1
                    )

            # Checks agent's limits
            a.clip_by_bound()

            # Calculates the fitness for the temporary position
            a.fit = function(a.position)

            # If new fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Copies its position and fitness to the agent
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

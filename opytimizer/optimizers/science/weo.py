"""Water Evaporation Optimization.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class WEO(Optimizer):
    """A WEO class, inherited from Optimizer.

    This is the designed class to define WEO-related
    variables and methods.

    References:
        A. Kaveh and T. Bakhshpoori.
        Water Evaporation Optimization: A novel physically inspired optimization algorithm.
        Computers & Structures (2016).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> WEO.")

        # Overrides its parent class with the receiving params
        super(WEO, self).__init__()

        # Minimum substrate energy
        self.E_min = -3.5

        # Maximum substrate energy
        self.E_max = -0.5

        # Minimum contact angle
        self.theta_min = -np.pi / 3.6

        # Maximum contact angle
        self.theta_max = -np.pi / 9

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def E_min(self) -> float:
        """Minimum substrate energy."""

        return self._E_min

    @E_min.setter
    def E_min(self, E_min: float) -> None:
        if not isinstance(E_min, (float, int)):
            raise e.TypeError("`E_min` should be a float or integer")

        self._E_min = E_min

    @property
    def E_max(self) -> float:
        """Maximum substrate energy."""

        return self._E_max

    @E_max.setter
    def E_max(self, E_max: float) -> None:
        if not isinstance(E_max, (float, int)):
            raise e.TypeError("`E_max` should be a float or integer")
        if E_max < self.E_min:
            raise e.ValueError("`E_max` should be >= `E_min`")

        self._E_max = E_max

    @property
    def theta_min(self) -> float:
        """Minimum contact angle."""

        return self._theta_min

    @theta_min.setter
    def theta_min(self, theta_min: float) -> None:
        if not isinstance(theta_min, (float, int)):
            raise e.TypeError("`theta_min` should be a float or integer")

        self._theta_min = theta_min

    @property
    def theta_max(self) -> float:
        """Maximum contact angle."""

        return self._theta_max

    @theta_max.setter
    def theta_max(self, theta_max: float) -> None:
        if not isinstance(theta_max, (float, int)):
            raise e.TypeError("`theta_max` should be a float or integer")
        if theta_max < self.theta_min:
            raise e.ValueError("`theta_max` should be >= `theta_min`")

        self._theta_max = theta_max

    def _evaporation_flux(self, theta: float) -> float:
        """Calculates the evaporation flux (eq. 7).

        Args:
            theta: Radian-based angle.

        Returns:
            (float): Evaporation flux.

        """

        # Calculates the evaporation flux (eq. 7)
        J = (
            (1 / 2.6)
            * ((2 / 3 + np.cos(theta) ** 3 / 3 - np.cos(theta)) ** (-2 / 3))
            * (1 - np.cos(theta))
        )

        return J

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Water Evaporation Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Sorts agents
        space.agents.sort(key=lambda x: x.fit)

        # Gathers best and worst agents
        best, worst = space.agents[0], space.agents[-1]

        # Iterates through all agents
        for agent in space.agents:
            # Makes a deep copy of current agent
            a = copy.deepcopy(agent)

            # Checks whether it is the first half of iterations
            if int(iteration <= n_iterations / 2):
                # Calculates the substrate energy (eq. 5)
                E_sub = ((self.E_max - self.E_min) * (a.fit - best.fit)) / (
                    worst.fit - best.fit + c.EPSILON
                ) + self.E_min

                # Calculates the Monolayer Evaporation Probability matrix (eq. 6)
                r1 = r.generate_uniform_random_number(
                    size=(agent.n_variables, agent.n_dimensions)
                )
                MEP = np.where(r1 < np.exp(E_sub), 1, 0)

                # Generates the step size (eq. 10)
                r2 = r.generate_uniform_random_number()
                i = r.generate_integer_random_number(0, space.n_agents)
                j = r.generate_integer_random_number(0, space.n_agents, i)
                S = r2 * (space.agents[i].position - space.agents[j].position)

                # Updates the agent's position (eq. 11)
                a.position += S * MEP

            # If it is the second half of iterations
            else:
                # Calculates the contact angle (eq. 8)
                theta = ((self.theta_max - self.theta_min) * (a.fit - best.fit)) / (
                    worst.fit - best.fit + c.EPSILON
                ) + self.theta_min

                # Calculates the Droplet Evaporation Probability matrix (eq. 9)
                r1 = r.generate_uniform_random_number(
                    size=(a.n_variables, a.n_dimensions)
                )
                DEP = np.where(r1 < self._evaporation_flux(theta), 1, 0)

                # Generates the step size (eq. 10)
                r2 = r.generate_uniform_random_number()
                i = r.generate_integer_random_number(0, space.n_agents)
                j = r.generate_integer_random_number(0, space.n_agents, i)
                S = r2 * (space.agents[i].position - space.agents[j].position)

                # Updates the agent's position (eq. 11)
                a.position += S * DEP

            # Checks agent's limits
            a.clip_by_bound()

            # Re-evaluates the temporary agent
            a.fit = function(a.position)

            # If temporary agent's fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Replace its position and fitness
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

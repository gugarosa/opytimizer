"""Water Evaporation Optimization."""

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.space import Space


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

        super(WEO, self).__init__()

        self.E_min = -3.5
        self.E_max = -0.5

        self.theta_min = -np.pi / 3.6
        self.theta_max = -np.pi / 9

        self.build(params)

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
        self, space: Space, function: Callable, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Water Evaporation Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        space.agents.sort(key=lambda x: x.fit)
        best, worst = space.agents[0], space.agents[-1]

        for agent in space.agents:
            a = copy.deepcopy(agent)

            if int(iteration <= n_iterations / 2):
                # Calculates the substrate energy (eq. 5)
                E_sub = ((self.E_max - self.E_min) * (a.fit - best.fit)) / (
                    worst.fit - best.fit + c.EPSILON
                ) + self.E_min

                # Calculates the Monolayer Evaporation Probability matrix (eq. 6)
                r1 = np.random.uniform(
                    0.0, 1.0, (agent.n_variables, agent.n_dimensions)
                )
                MEP = np.where(r1 < np.exp(E_sub), 1, 0)

                # Generates the step size (eq. 10)
                r2 = np.random.uniform(0.0, 1.0, 1)
                i = np.random.randint(0, space.n_agents, None)
                j = r.integer(0, space.n_agents, exclude=i, size=None)
                S = r2 * (space.agents[i].position - space.agents[j].position)

                # Updates the agent's position (eq. 11)
                a.position += S * MEP
            else:
                # Calculates the contact angle (eq. 8)
                theta = ((self.theta_max - self.theta_min) * (a.fit - best.fit)) / (
                    worst.fit - best.fit + c.EPSILON
                ) + self.theta_min

                # Calculates the Droplet Evaporation Probability matrix (eq. 9)
                r1 = np.random.uniform(0.0, 1.0, (a.n_variables, a.n_dimensions))
                DEP = np.where(r1 < self._evaporation_flux(theta), 1, 0)

                # Generates the step size (eq. 10)
                r2 = np.random.uniform(0.0, 1.0, 1)
                i = np.random.randint(0, space.n_agents, None)
                j = r.integer(0, space.n_agents, exclude=i, size=None)
                S = r2 * (space.agents[i].position - space.agents[j].position)

                # Updates the agent's position (eq. 11)
                a.position += S * DEP
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

"""Interactive Search Algorithm.
"""

import copy
import time
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


class ISA(Optimizer):
    """An ISA class, inherited from Optimizer.

    This is the designed class to define ISA-related
    variables and methods.

    References:
        A. Mortazavi, V. Toğan and A. Nuhoğlu.
        Interactive search algorithm: A new hybrid metaheuristic optimization algorithm.
        Engineering Applications of Artificial Intelligence (2018).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> ISA.")

        super(ISA, self).__init__()

        self.w = 0.7
        self.tau = 0.3

        self.build(params)

        logger.info("Class overrided.")

    @property
    def w(self) -> float:
        """Inertia weight."""

        return self._w

    @w.setter
    def w(self, w: float) -> None:
        if not isinstance(w, (float, int)):
            raise e.TypeError("`w` should be a float or integer")
        if w < 0:
            raise e.ValueError("`w` should be >= 0")

        self._w = w

    @property
    def tau(self) -> float:
        """Tendency factor."""

        return self._tau

    @tau.setter
    def tau(self, tau: float) -> None:
        if not isinstance(tau, (float, int)):
            raise e.TypeError("`tau` should be a float or integer")
        if tau < 0:
            raise e.ValueError("`tau` should be >= 0")

        self._tau = tau

    @property
    def local_position(self) -> np.ndarray:
        """Array of velocities."""

        return self._local_position

    @local_position.setter
    def local_position(self, local_position: np.ndarray) -> None:
        if not isinstance(local_position, np.ndarray):
            raise e.TypeError("`local_position` should be a numpy array")

        self._local_position = local_position

    @property
    def velocity(self) -> np.ndarray:
        """Array of velocities."""

        return self._velocity

    @velocity.setter
    def velocity(self, velocity: np.ndarray) -> None:
        if not isinstance(velocity, np.ndarray):
            raise e.TypeError("`velocity` should be a numpy array")

        self._velocity = velocity

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def evaluate(self, space: Space, function: Function) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A Space object that will be evaluated.
            function: A Function object that will be used as the objective function.

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

    def update(self, space: Space, function: Function) -> None:
        """Wraps Interactive Search Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        space.agents.sort(key=lambda x: x.fit)
        best, worst = space.agents[0], space.agents[-1]

        coef = [
            (best.fit - agent.fit) / (best.fit - worst.fit + c.EPSILON)
            for agent in space.agents
        ]
        w_coef = [cf / (np.sum(coef) + c.EPSILON) for cf in coef]

        w_position = np.sum(
            [cf * agent.position for cf, agent in zip(w_coef, space.agents)], axis=0
        )
        w_fit = function(w_position)

        for i, agent in enumerate(space.agents):
            r1 = r.generate_uniform_random_number()
            idx = r.generate_integer_random_number(high=space.n_agents, exclude_value=i)

            if r1 >= self.tau:
                phi3 = r.generate_uniform_random_number()
                phi2 = 2 * r.generate_uniform_random_number()
                phi1 = -(phi2 + phi3) * r.generate_uniform_random_number()

                # Updates the agent's velocity (eq. 6.1)
                self.velocity[i] = (
                    self.w * self.velocity[i]
                    + phi1 * (self.local_position[idx] - agent.position)
                    + phi2 * (space.best_agent.position - self.local_position[idx])
                    + phi3 * (w_position - self.local_position[idx])
                )
            else:
                r2 = r.generate_uniform_random_number()
                if agent.fit < space.agents[idx].fit:
                    # Updates agent's velocity (eq. 6.2 - top)
                    self.velocity[i] = r2 * (
                        agent.position - space.agents[idx].position
                    )
                else:
                    # Updates agent's velocity (eq. 6.2 - bottom)
                    self.velocity[i] = r2 * (
                        space.agents[idx].position - agent.position
                    )

            # Updates agent's position and clip its bounds (eq. 6.3)
            agent.position += self.velocity[i]
            agent.clip_by_bound()

            agent.fit = function(agent.position)
            local_fit = function(self.local_position[i])

            if w_fit < agent.fit:
                if w_fit < local_fit:
                    self.local_position[i] = copy.deepcopy(w_position)
            else:
                if agent.fit < local_fit:
                    self.local_position[i] = copy.deepcopy(agent.position)

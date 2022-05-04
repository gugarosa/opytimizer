"""Tug Of War Optimization.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class TWO(Optimizer):
    """A TWO class, inherited from Optimizer.

    This is the designed class to define TWO-related
    variables and methods.

    References:
        A. Kaveh. Tug of War Optimization.
        Advances in Metaheuristic Algorithms for Optimal Design of Structures (2016).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> TWO.")

        # Overrides its parent class with the receiving params
        super(TWO, self).__init__()

        # Static friction coefficient
        self.mu_s = 1

        # Kinematic friction coefficient
        self.mu_k = 1

        # Time displacement
        self.delta_t = 1

        # Speed constant
        self.alpha = 0.9

        # Scaling factor
        self.beta = 0.05

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def mu_s(self) -> float:
        """Static friction coefficient."""

        return self._mu_s

    @mu_s.setter
    def mu_s(self, mu_s: float) -> None:
        if not isinstance(mu_s, (float, int)):
            raise e.TypeError("`mu_s` should be a float or integer")
        if mu_s < 0:
            raise e.ValueError("`mu_s` should be >= 0")

        self._mu_s = mu_s

    @property
    def mu_k(self) -> float:
        """Kinematic friction coefficient."""

        return self._mu_k

    @mu_k.setter
    def mu_k(self, mu_k: float) -> None:
        if not isinstance(mu_k, (float, int)):
            raise e.TypeError("`mu_k` should be a float or integer")
        if mu_k < 0:
            raise e.ValueError("`mu_k` should be >= 0")

        self._mu_k = mu_k

    @property
    def delta_t(self) -> float:
        """Time displacement."""

        return self._delta_t

    @delta_t.setter
    def delta_t(self, delta_t: float) -> None:
        if not isinstance(delta_t, (float, int)):
            raise e.TypeError("`delta_t` should be a float or integer")
        if delta_t < 0:
            raise e.ValueError("`delta_t` should be >= 0")

        self._delta_t = delta_t

    @property
    def alpha(self) -> float:
        """Speed constant."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        if alpha < 0.9 or alpha > 1:
            raise e.ValueError("`alpha` should be between 0.9 and 1")

        self._alpha = alpha

    @property
    def beta(self) -> float:
        """Scaling factor."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        if beta <= 0 or beta > 1:
            raise e.ValueError("`beta` should be greater than 0 and less than 1")

        self._beta = beta

    def _constraint_handle(
        self, agents: List[Agent], best_agent: Agent, function: Function, iteration: int
    ) -> None:
        """Performs the constraint handling procedure (eq. 11).

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.

        """

        # Iterates through every agent
        for agent in agents:
            # Generates a random number
            r1 = r.generate_uniform_random_number()

            # If random is smaller than 0.5
            if r1 < 0.5:
                # Generates a gaussian random number
                r2 = r.generate_gaussian_random_number()

                # Updates the agent's position
                agent.position = best_agent.position + (r2 / iteration) * (
                    best_agent.position - agent.position
                )

            # Clips its limits
            agent.clip_by_bound()

            # Re-calculates its fitness
            agent.fit = function(agent.position)

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Tug of War Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Sorts agents
        space.agents.sort(key=lambda x: x.fit)

        # Gathers best and worst fitness
        best_fit, worst_fit = space.agents[0].fit, space.agents[-1].fit

        # Calculates the agents' weights
        weights = [
            (agent.fit - worst_fit) / (best_fit - worst_fit + c.EPSILON) + 1
            for agent in space.agents
        ]

        # We copy a temporary list for iterating purposes
        temp_agents = copy.deepcopy(space.agents)

        # Linearly decreasing `mu_k`
        mu_k = self.mu_k - (self.mu_k - 0.1) * (iteration / n_iterations)

        # Iterates through 'i' agents
        for i, temp1 in enumerate(temp_agents):
            # Initializes `delta` as zero
            delta = 0.0

            # Iterates through 'j' agents
            for j, temp2 in enumerate(temp_agents):
                # If weight from agent `i` is smaller than weight from agent `j`
                if weights[i] < weights[j]:
                    # Calculates the residual force (eq. 6)
                    force = (
                        np.maximum(weights[i] * self.mu_s, weights[j] * self.mu_s)
                        - weights[i] * mu_k
                    )

                    # Calculates the gravitational acceleration (eq. 8)
                    g = temp2.position - temp1.position

                    # Calculates the acceleration (eq. 7)
                    acceleration = (force / (weights[i] * mu_k)) * g

                    # Generates a random gaussian number
                    r1 = r.generate_gaussian_random_number(
                        size=(temp1.n_variables, temp1.n_dimensions)
                    )

                    # Calculates the displacement (eq. 9-10)
                    delta += 0.5 * acceleration * self.delta_t**2 + np.multiply(
                        self.alpha**iteration
                        * self.beta
                        * (np.expand_dims(temp1.ub, -1) - np.expand_dims(temp1.lb, -1)),
                        r1,
                    )

            # Updates the temporary agent's position (eq. 11)
            temp1.position += delta

        # Performs the constraint handling
        self._constraint_handle(temp_agents, space.best_agent, function, iteration + 1)

        # Iterates through real and temporary populations
        for agent, temp in zip(space.agents, temp_agents):
            # If temporary agent is better than real one
            if temp.fit < agent.fit:
                # Updates its position and fitness
                agent.position = copy.deepcopy(temp.position)
                agent.fit = copy.deepcopy(temp.fit)

"""Tug Of War Optimization."""

import copy
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


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

        super(TWO, self).__init__()

        self.mu_s = 1
        self.mu_k = 1
        self.delta_t = 1

        self.alpha = 0.9
        self.beta = 0.05

        self.build(params)

    def _constraint_handle(
        self, agents: List[Agent], best_agent: Agent, function: Callable, iteration: int
    ) -> None:
        """Performs the constraint handling procedure (eq. 11).

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function: A callable that will be used as the objective function.
            iteration: Current iteration.

        """

        for agent in agents:
            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 < 0.5:
                r2 = np.random.normal(0.0, 1.0, 1)

                agent.position = best_agent.position + (r2 / iteration) * (
                    best_agent.position - agent.position
                )
            agent.clip_by_bound()

            agent.fit = function(agent.position)

    def update(
        self, space: Space, function: Callable, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Tug of War Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        space.agents.sort(key=lambda x: x.fit)
        best_fit, worst_fit = space.agents[0].fit, space.agents[-1].fit

        weights = [
            (agent.fit - worst_fit) / (best_fit - worst_fit + c.EPSILON) + 1
            for agent in space.agents
        ]

        temp_agents = copy.deepcopy(space.agents)

        mu_k = self.mu_k - (self.mu_k - 0.1) * (iteration / n_iterations)

        for i, temp1 in enumerate(temp_agents):
            delta = 0.0

            for j, temp2 in enumerate(temp_agents):
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

                    r1 = np.random.normal(
                        0.0, 1.0, (temp1.n_variables, temp1.n_dimensions)
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

        self._constraint_handle(temp_agents, space.best_agent, function, iteration + 1)

        for agent, temp in zip(space.agents, temp_agents):
            if temp.fit < agent.fit:
                agent.position = copy.deepcopy(temp.position)
                agent.fit = copy.deepcopy(temp.fit)

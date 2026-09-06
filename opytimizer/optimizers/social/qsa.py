# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Queuing Search Algorithm.

References:
    J. Zhang et al. Queuing search algorithm: A novel metaheuristic algorithm
    for solving engineering optimization problems.
    Applied Mathematical Modelling (2018).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class QSA(Optimizer):
    """Implement Queuing Search Algorithm.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the three queue-search business phases.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            This optimizer has no algorithm-specific configuration keys.
            Queue sizes use reciprocal leader fitness when the best fitness exceeds epsilon,
            otherwise the population is divided equally among the three queues.

        """

        super(QSA, self).__init__()

        self.build(params)

    def _calculate_queue(self, n_agents: int, t_1: float, t_2: float, t_3: float) -> tuple[int, int, int]:
        if t_1 > c.EPSILON:
            n_1 = (1 / t_1) / ((1 / t_1) + (1 / t_2) + (1 / t_3))
            n_2 = (1 / t_2) / ((1 / t_1) + (1 / t_2) + (1 / t_3))
            n_3 = (1 / t_3) / ((1 / t_1) + (1 / t_2) + (1 / t_3))
        else:
            n_1 = 1 / 3
            n_2 = 1 / 3
            n_3 = 1 / 3

        q_1 = int(n_1 * n_agents)
        q_2 = int(n_2 * n_agents)
        q_3 = int(n_3 * n_agents)

        return q_1, q_2, q_3

    def _business_one(self, agents: list[Agent], function: Callable, beta: float) -> None:
        agents.sort(key=lambda x: x.fit)

        A_1, A_2, A_3 = (
            copy.deepcopy(agents[0]),
            copy.deepcopy(agents[1]),
            copy.deepcopy(agents[2]),
        )

        q_1, q_2, _ = self._calculate_queue(len(agents), A_1.fit, A_2.fit, A_3.fit)

        # Represents the update patterns by eq. 4 and eq. 5
        case = None

        for i, agent in enumerate(agents):
            a = copy.deepcopy(agent)

            if i < q_1:
                if i == 0:
                    case = 1

                A = copy.deepcopy(A_1)
            elif q_1 <= i < q_1 + q_2:
                if i == q_1:
                    case = 1

                A = copy.deepcopy(A_2)
            else:
                if i == q_1 + q_2:
                    case = 1

                A = copy.deepcopy(A_3)

            alpha = np.random.uniform(-1, 1, 1)
            E = np.random.gamma(1, 0.5, (agent.n_variables, agent.n_dimensions))

            if case == 1:
                e = np.random.gamma(1, 0.5, 1)

                # Calculates the fluctuation (eq. 6)
                F_1 = beta * alpha * (E * np.fabs(A.position - a.position)) + e * (A.position - a.position)

                # Updates the temporary agent's position (eq. 4)
                a.position = A.position + F_1

                a.fit = function(a.position)
                if a.fit < agent.fit:
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

                    case = 1
                else:
                    case = 2
            else:
                # Calculates the fluctuation (eq. 7)
                F_2 = beta * alpha * (E * np.fabs(A.position - a.position))

                # Updates the temporary agent's position (eq. 5)
                a.position += F_2

                a.fit = function(a.position)
                if a.fit < agent.fit:
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

                    case = 2
                else:
                    case = 1

    def _business_two(self, agents: list[Agent], function: Callable) -> None:
        agents.sort(key=lambda x: x.fit)

        A_1, A_2, A_3 = (
            copy.deepcopy(agents[0]),
            copy.deepcopy(agents[1]),
            copy.deepcopy(agents[2]),
        )

        q_1, q_2, _ = self._calculate_queue(len(agents), A_1.fit, A_2.fit, A_3.fit)

        pr = [i / len(agents) for i in range(1, len(agents) + 1)]

        cv = A_1.fit / (A_2.fit + A_3.fit + c.EPSILON)

        for i, agent in enumerate(agents):
            a = copy.deepcopy(agent)

            if i < q_1:
                A = copy.deepcopy(A_1)
            elif q_1 <= i < q_1 + q_2:
                A = copy.deepcopy(A_2)
            else:
                A = copy.deepcopy(A_3)

            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 < pr[i]:
                A_1, A_2 = np.random.choice(agents, 2, replace=False)

                r2 = np.random.uniform(0.0, 1.0, 1)
                e = np.random.gamma(1, 0.5, 1)

                if r2 < cv:
                    # Calculates the fluctuation (eq. 14)
                    F_1 = e * (A_1.position - A_2.position)

                    # Update agent's position (eq. 12)
                    a.position += F_1
                else:
                    # Calculates the fluctuation (eq. 15)
                    F_2 = e * (A.position - A_1.position)

                    # Update agent's position (eq. 13)
                    a.position += F_2

                a.fit = function(a.position)
                if a.fit < agent.fit:
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

    def _business_three(self, agents: list[Agent], function: Callable) -> None:
        agents.sort(key=lambda x: x.fit)

        pr = [i / len(agents) for i in range(1, len(agents) + 1)]

        for i, agent in enumerate(agents):
            a = copy.deepcopy(agent)

            for j in range(agent.n_variables):
                r1 = np.random.uniform(0.0, 1.0, 1)
                if r1 < pr[i]:
                    A_1, A_2 = np.random.choice(agents, 2, replace=False)
                    e = np.random.gamma(1, 0.5, 1)

                    # Updates temporary agent's position (eq. 17)
                    a.position[j] = A_1.position[j] + e * (A_2.position[j] - a.position[j])

                a.fit = function(a.position)
                if a.fit < agent.fit:
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

    def update(self, space: Space, function: Callable, iteration: int, n_iterations: int) -> None:
        beta = np.exp(np.log(1 / (iteration + c.EPSILON)) * np.sqrt(iteration / n_iterations))

        self._business_one(space.agents, function, beta)
        self._business_two(space.agents, function)
        self._business_three(space.agents, function)

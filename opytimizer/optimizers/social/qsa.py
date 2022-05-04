"""Queuing Search Algorithm.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class QSA(Optimizer):
    """A QSA class, inherited from Optimizer.

    This is the designed class to define QSA-related
    variables and methods.

    References:
        J. Zhang et al. Queuing search algorithm: A novel metaheuristic algorithm
        for solving engineering optimization problems.
        Applied Mathematical Modelling (2018).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> QSA.")

        # Overrides its parent class with the receiving params
        super(QSA, self).__init__()

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    def _calculate_queue(
        self, n_agents: int, t_1: float, t_2: float, t_3: float
    ) -> Tuple[int, int, int]:
        """Calculates the number of agents that belongs to each queue.

        Args:
            n_agents: Number of agents.
            t_1: Fitness value of first agent in the population.
            t_2: Fitness value of second agent in the population.
            t_3: Fitness value of third agent in the population.

        Returns:
            (Tuple[int, int, int]): The number of agents in first, second and third queues.

        """

        # Checks if potential service time is bigger than `epsilon`
        if t_1 > c.EPSILON:
            # Calculates the proportion of agents in first, second and third queues
            n_1 = (1 / t_1) / ((1 / t_1) + (1 / t_2) + (1 / t_3))
            n_2 = (1 / t_2) / ((1 / t_1) + (1 / t_2) + (1 / t_3))
            n_3 = (1 / t_3) / ((1 / t_1) + (1 / t_2) + (1 / t_3))

        # If the potential service time is smaller than `epsilon`
        else:
            # Each queue will have 1/3 ratio
            n_1 = 1 / 3
            n_2 = 1 / 3
            n_3 = 1 / 3

        # Calculates the number of agents that belongs to each queue
        q_1 = int(n_1 * n_agents)
        q_2 = int(n_2 * n_agents)
        q_3 = int(n_3 * n_agents)

        return q_1, q_2, q_3

    def _business_one(
        self, agents: List[Agent], function: Function, beta: float
    ) -> None:
        """Performs the first business phase.

        Args:
            agents: List of agents.
            function: A Function object that will be used as the objective function.
            beta: Range of fluctuation.

        """

        # Sorts agents
        agents.sort(key=lambda x: x.fit)

        # Copies temporary agents to represent `A_1`, `A_2` and `A_3`
        A_1, A_2, A_3 = (
            copy.deepcopy(agents[0]),
            copy.deepcopy(agents[1]),
            copy.deepcopy(agents[2]),
        )

        # Calculates the number of agents in each queue
        q_1, q_2, _ = self._calculate_queue(len(agents), A_1.fit, A_2.fit, A_3.fit)

        # Represents the update patterns by eq. 4 and eq. 5
        case = None

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Creates another temporary agent
            a = copy.deepcopy(agent)

            # If index is smaller than the number of agents in first queue
            if i < q_1:
                # If it is the first agent in first queue
                if i == 0:
                    # Defines the case as one
                    case = 1

                # `A` will receive a copy from `A_1`
                A = copy.deepcopy(A_1)

            # If index is between first and second queues
            elif q_1 <= i < q_1 + q_2:
                # If index is the first agent in second queue
                if i == q_1:
                    # Defines the case as one
                    case = 1

                # `A` will receive a copy from `A_2`
                A = copy.deepcopy(A_2)

            # If index is between second and third queues
            else:
                # If index is the first agent in third queue
                if i == q_1 + q_2:
                    # Defines the case as one
                    case = 1

                # `A` will receive a copy from `A_3`
                A = copy.deepcopy(A_3)

            # Generates a uniform random number
            alpha = r.generate_uniform_random_number(-1, 1)

            # Generates an Erlang distribution
            E = r.generate_gamma_random_number(
                1, 0.5, (agent.n_variables, agent.n_dimensions)
            )

            # If case is defined as one
            if case == 1:
                # Generates an Erlang number
                e = r.generate_gamma_random_number(1, 0.5, 1)

                # Calculates the fluctuation (eq. 6)
                F_1 = beta * alpha * (E * np.fabs(A.position - a.position)) + e * (
                    A.position - a.position
                )

                # Updates the temporary agent's position (eq. 4)
                a.position = A.position + F_1

                # Evaluates the agent
                a.fit = function(a.position)

                # If new fitness is better than current agent's fitness
                if a.fit < agent.fit:
                    # Replaces the current agent's position and fitness
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

                    # Defines the case as one
                    case = 1

                # If new fitness is worse than current agent's fitness
                else:
                    # Defines the case as two
                    case = 2

            # If case is defined as two
            else:
                # Calculates the fluctuation (eq. 7)
                F_2 = beta * alpha * (E * np.fabs(A.position - a.position))

                # Updates the temporary agent's position (eq. 5)
                a.position += F_2

                # Evaluates the agent
                a.fit = function(a.position)

                # If new fitness is better than current agent's fitness
                if a.fit < agent.fit:
                    # Replaces the current agent's position and fitness
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

                    # Defines the case as two
                    case = 2

                # If new fitness is worse than current agent's fitness
                else:
                    # Defines the case as one
                    case = 1

    def _business_two(self, agents: List[Agent], function: Function) -> None:
        """Performs the second business phase.

        Args:
            agents: List of agents.
            function: A Function object that will be used as the objective function.

        """

        # Sorts agents
        agents.sort(key=lambda x: x.fit)

        # Copies temporary agents to represent `A_1`, `A_2` and `A_3`
        A_1, A_2, A_3 = (
            copy.deepcopy(agents[0]),
            copy.deepcopy(agents[1]),
            copy.deepcopy(agents[2]),
        )

        # Calculates the number of agents in each queue
        q_1, q_2, _ = self._calculate_queue(len(agents), A_1.fit, A_2.fit, A_3.fit)

        # Calculates the probability of handling the business
        pr = [i / len(agents) for i in range(1, len(agents) + 1)]

        # Calculates the confusion degree
        cv = A_1.fit / (A_2.fit + A_3.fit + c.EPSILON)

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Creates another temporary agent
            a = copy.deepcopy(agent)

            # If index is smaller than the number of agents in first queue
            if i < q_1:
                # `A` will receive a copy from `A_1`
                A = copy.deepcopy(A_1)

            # If index is between first and second queues
            elif q_1 <= i < q_1 + q_2:
                # `A` will receive a copy from `A_2`
                A = copy.deepcopy(A_2)

            # If index is between second and third queues
            else:
                # `A` will receive a copy from `A_3`
                A = copy.deepcopy(A_3)

            # Generates a uniform random number
            r1 = r.generate_uniform_random_number()

            # If random number is smaller than probability of handling the business
            if r1 < pr[i]:
                # Randomly selects two individuals
                A_1, A_2 = np.random.choice(agents, 2, replace=False)

                # Generates another uniform random number
                r2 = r.generate_uniform_random_number()

                # Generates an Erlang number
                e = r.generate_gamma_random_number(1, 0.5, 1)

                # If random number is smaller than confusion degree
                if r2 < cv:
                    # Calculates the fluctuation (eq. 14)
                    F_1 = e * (A_1.position - A_2.position)

                    # Update agent's position (eq. 12)
                    a.position += F_1

                # If random number is bigger than confusion degree
                else:
                    # Calculates the fluctuation (eq. 15)
                    F_2 = e * (A.position - A_1.position)

                    # Update agent's position (eq. 13)
                    a.position += F_2

                # Evaluates the agent
                a.fit = function(a.position)

                # If the new fitness is better than the current agent's fitness
                if a.fit < agent.fit:
                    # Replaces the current agent's position and fitness
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

    def _business_three(self, agents: List[Agent], function: Function) -> None:
        """Performs the third business phase.

        Args:
            agents: List of agents.
            function: A Function object that will be used as the objective function.

        """

        # Sorts agents
        agents.sort(key=lambda x: x.fit)

        # Calculates the probability of handling the business
        pr = [i / len(agents) for i in range(1, len(agents) + 1)]

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Creates another temporary agent
            a = copy.deepcopy(agent)

            # Iterates through all decision variables
            for j in range(agent.n_variables):
                # Generates a uniform random number
                r1 = r.generate_uniform_random_number()

                # If random number is smaller than probability of handling the business
                if r1 < pr[i]:
                    # Randomly selects two individuals
                    A_1, A_2 = np.random.choice(agents, 2, replace=False)

                    # Generates an Erlang number
                    e = r.generate_gamma_random_number(1, 0.5, 1)

                    # Updates temporary agent's position (eq. 17)
                    a.position[j] = A_1.position[j] + e * (
                        A_2.position[j] - a.position[j]
                    )

                # Evaluates the agent
                a.fit = function(a.position)

                # If the new fitness is better than the current agent's fitness
                if a.fit < agent.fit:
                    # Replaces the current agent's position and fitness
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Queue Search Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Calculates the range of fluctuation.
        beta = np.exp(
            np.log(1 / (iteration + c.EPSILON)) * np.sqrt(iteration / n_iterations)
        )

        # Performs the first business phase
        self._business_one(space.agents, function, beta)

        # Performs the second business phase
        self._business_two(space.agents, function)

        # Performs the third business phase
        self._business_three(space.agents, function)

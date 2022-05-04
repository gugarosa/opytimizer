"""Artificial Ecosystem-based Optimization.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.random as r
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class AEO(Optimizer):
    """An AEO class, inherited from Optimizer.

    This is the designed class to define AEO-related
    variables and methods.

    References:
        W. Zhao, L. Wang and Z. Zhang.
        Artificial ecosystem-based optimization: a novel nature-inspired meta-heuristic algorithm.
        Neural Computing and Applications (2019).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        # Overrides its parent class with the receiving params
        super(AEO, self).__init__()

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    def _production(
        self, agent: Agent, best_agent: Agent, iteration: int, n_iterations: int
    ) -> Agent:
        """Performs the producer update (eq. 1).

        Args:
            agent: Current agent.
            best_agent: Best agent.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        Returns:
            (Agent): An updated producer.

        """

        # Makes a deep copy of agent
        a = copy.deepcopy(agent)

        # Calculates the alpha factor (eq. 2)
        alpha = (1 - iteration / n_iterations) * r.generate_uniform_random_number()

        # For every possible decision variable
        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            # Updates its position
            a.position[j] = (1 - alpha) * best_agent.position[
                j
            ] + alpha * r.generate_uniform_random_number(lb, ub, a.n_dimensions)

        return a

    def _herbivore_consumption(self, agent: Agent, producer: Agent, C: float) -> Agent:
        """Performs the consumption update by a herbivore (eq. 6).

        Args:
            agent: Current agent.
            producer: Producer agent.
            C: Consumption factor.

        Returns:
            An updated consumption by a herbivore.

        """

        # Makes a deep copy of agent
        a = copy.deepcopy(agent)

        # Updates its position
        a.position += C * (agent.position - producer.position)

        return a

    def _omnivore_consumption(
        self, agent: Agent, producer: Agent, consumer: Agent, C: float
    ) -> Agent:
        """Performs the consumption update by an omnivore (eq. 8)

        Args:
            agent: Current agent.
            producer: Producer agent.
            consumer: Consumer agent.
            C: Consumption factor.

        Returns:
            (Agent): An updated consumption by an omnivore.

        """

        # Makes a deep copy of agent
        a = copy.deepcopy(agent)

        # Generates the second random number
        r2 = r.generate_uniform_random_number()

        # Updates its position
        a.position += C * r2 * (a.position - producer.position) + (1 - r2) * (
            a.position - consumer.position
        )

        return a

    def _carnivore_consumption(self, agent: Agent, consumer: Agent, C: float) -> Agent:
        """Performs the consumption update by a carnivore (eq. 7).

        Args:
            agent: Current agent.
            consumer: Consumer agent.
            C: Consumption factor.

        Returns:
            (Agent): An updated consumption by a carnivore.

        """

        # Makes a deep copy of agent
        a = copy.deepcopy(agent)

        # Updates its position
        a.position += C * (a.position - consumer.position)

        return a

    def _update_composition(
        self,
        agents: List[Agent],
        best_agent: Agent,
        function: Function,
        iteration: int,
        n_iterations: int,
    ) -> None:
        """Wraps production and consumption updates over all
        agents and variables (eq. 1-8).

        Args:
            agents: List of agents.
            best_agent: Global best agent.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Sorts agents according to their energy
        agents.sort(key=lambda x: x.fit, reverse=True)

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # If it is the first agent
            if i == 0:
                # It will surely be a producer
                a = self._production(agent, best_agent, iteration, n_iterations)

            # If it is not the first agent
            else:
                # Generates the first random number
                r1 = r.generate_uniform_random_number()

                # Generates gaussian random numbers
                v1 = r.generate_gaussian_random_number()
                v2 = r.generate_gaussian_random_number()

                # Calculates the consumption factor (eq. 4)
                C = 0.5 * v1 / np.abs(v2)

                # If random number lies in the first third
                if r1 < 1 / 3:
                    # It will surely be a herbivore
                    a = self._herbivore_consumption(agent, agents[0], C)

                # If random number lies in the second third
                elif 1 / 3 <= r1 <= 2 / 3:
                    # Generates a random index from the population
                    j = int(r.generate_uniform_random_number(1, i))

                    # It will surely be a omnivore
                    a = self._omnivore_consumption(agent, agents[0], agents[j], C)

                # If random number lies in the last third
                else:
                    # Generates a random index from the population
                    j = int(r.generate_uniform_random_number(1, i))

                    # It will surely be a carnivore
                    a = self._carnivore_consumption(agent, agents[j], C)

            # Checks agent's limits
            a.clip_by_bound()

            # Calculates the fitness for the temporary position
            a.fit = function(a.position)

            # If new fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Copies its position and fitness to the agent
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

    def _update_decomposition(
        self, agents: List[Agent], best_agent: Agent, function: Function
    ) -> None:
        """Wraps decomposition updates over all
        agents and variables (eq. 9).

        Args:
            agents: List of agents.
            best_agent: Global best agent.
            function: A Function object that will be used as the objective function.

        """

        # Iterates through all agents
        for agent in agents:
            # Makes a deep copy of current agent
            a = copy.deepcopy(agent)

            # Calculates the decomposition factor (eq. 10)
            D = 3 * r.generate_gaussian_random_number()

            # Generates the third random number
            r3 = r.generate_uniform_random_number()

            # First weight coefficient (eq. 11)
            e = r3 * int(r.generate_uniform_random_number(1, 2)) - 1

            # Second weight coefficient (eq. 12)
            _h = 2 * r3 - 1

            # Updates the new agent position
            a.position = best_agent.position + D * (
                e * best_agent.position - _h * agent.position
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

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Artificial Ecosystem-based Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Updates agents within the composition step
        self._update_composition(
            space.agents, space.best_agent, function, iteration, n_iterations
        )

        # Updates agents within the decomposition step
        self._update_decomposition(space.agents, space.best_agent, function)

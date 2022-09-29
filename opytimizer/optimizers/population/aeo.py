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

        super(AEO, self).__init__()

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

        a = copy.deepcopy(agent)

        # Calculates the alpha factor (eq. 2)
        alpha = (1 - iteration / n_iterations) * r.generate_uniform_random_number()

        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
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

        a = copy.deepcopy(agent)
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

        a = copy.deepcopy(agent)

        r2 = r.generate_uniform_random_number()
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

        a = copy.deepcopy(agent)
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

        agents.sort(key=lambda x: x.fit, reverse=True)
        for i, agent in enumerate(agents):
            if i == 0:
                a = self._production(agent, best_agent, iteration, n_iterations)
            else:
                r1 = r.generate_uniform_random_number()

                v1 = r.generate_gaussian_random_number()
                v2 = r.generate_gaussian_random_number()

                # Calculates the consumption factor (eq. 4)
                C = 0.5 * v1 / np.abs(v2)

                if r1 < 1 / 3:
                    a = self._herbivore_consumption(agent, agents[0], C)
                elif 1 / 3 <= r1 <= 2 / 3:
                    j = int(r.generate_uniform_random_number(1, i))
                    a = self._omnivore_consumption(agent, agents[0], agents[j], C)
                else:
                    j = int(r.generate_uniform_random_number(1, i))
                    a = self._carnivore_consumption(agent, agents[j], C)

            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
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

        for agent in agents:
            a = copy.deepcopy(agent)

            # Calculates the decomposition factor (eq. 10)
            D = 3 * r.generate_gaussian_random_number()

            r3 = r.generate_uniform_random_number()

            # First weight coefficient (eq. 11)
            e = r3 * int(r.generate_uniform_random_number(1, 2)) - 1

            # Second weight coefficient (eq. 12)
            _h = 2 * r3 - 1

            a.position = best_agent.position + D * (
                e * best_agent.position - _h * agent.position
            )
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
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

        self._update_composition(
            space.agents, space.best_agent, function, iteration, n_iterations
        )
        self._update_decomposition(space.agents, space.best_agent, function)

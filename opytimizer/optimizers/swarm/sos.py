"""Symbiotic Organisms Search.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class SOS(Optimizer):
    """An SOS class, inherited from Optimizer.

    This is the designed class to define SOS-related
    variables and methods.

    References:
        M.-Y. Cheng and D. Prayogo. Symbiotic Organisms Search: A new metaheuristic optimization algorithm.
        Computers & Structures (2014).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> SOS.")

        # Overrides its parent class with the receiving params
        super(SOS, self).__init__()

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    def _mutualism(
        self, agent_i: Agent, agent_j: Agent, best_agent: Agent, function: Function
    ) -> None:
        """Performs the mutualism operation.

        Args:
            agent_i: Selected `i` agent.
            agent_j: Selected `j` agent.
            best_agent: Global best agent.
            function: A Function object that will be used as the objective function.

        """

        # Copies temporary agents from `i` and `j`
        a = copy.deepcopy(agent_i)
        b = copy.deepcopy(agent_j)

        # Calculates the mutual vector (eq. 3)
        mutual_vector = (agent_i.position + agent_j.position) / 2

        # Calculates the benefitial factors
        BF_1, BF_2 = np.random.choice([1, 2], 2, replace=False)

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number()

        # Re-calculates the new positions (eq. 1 and 2)
        a.position += r1 * (best_agent.position - mutual_vector * BF_1)
        b.position += r1 * (best_agent.position - mutual_vector * BF_2)

        # Checks their limits
        a.clip_by_bound()
        b.clip_by_bound()

        # Evaluates both agents
        a.fit = function(a.position)
        b.fit = function(b.position)

        # If new position is better than agent's `i` position
        if a.fit < agent_i.fit:
            # Replaces the agent's `i` position and fitness
            agent_i.position = copy.deepcopy(a.position)
            agent_i.fit = copy.deepcopy(a.fit)

        # If new position is better than agent's `j` position
        if b.fit < agent_j.fit:
            # Replaces the agent's `j` position and fitness
            agent_j.position = copy.deepcopy(b.position)
            agent_j.fit = copy.deepcopy(b.fit)

    def _commensalism(
        self, agent_i: Agent, agent_j: Agent, best_agent: Agent, function: Function
    ) -> None:
        """Performs the commensalism operation.

        Args:
            agent_i: Selected `i` agent.
            agent_j: Selected `j` agent.
            best_agent: Global best agent.
            function: A Function object that will be used as the objective function.

        """

        # Copies a temporary agent from `i`
        a = copy.deepcopy(agent_i)

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number(-1, 1)

        # Updates the agent's position (eq. 4)
        a.position += r1 * (best_agent.position - agent_j.position)

        # Checks its limits
        a.clip_by_bound()

        # Evaluates its new position
        a.fit = function(a.position)

        # If the new position is better than the current agent's position
        if a.fit < agent_i.fit:
            # Replaces the current agent's position and fitness
            agent_i.position = copy.deepcopy(a.position)
            agent_i.fit = copy.deepcopy(a.fit)

    def _parasitism(self, agent_i: Agent, agent_j: Agent, function: Function) -> None:
        """Performs the parasitism operation.

        Args:
            agent_i: Selected `i` agent.
            agent_j: Selected `j` agent.
            function: A Function object that will be used as the objective function.

        """

        # Creates a temporary parasite agent
        p = copy.deepcopy(agent_i)

        # Generates a integer random number
        r1 = r.generate_integer_random_number(0, agent_i.n_variables)

        # Updates its position on selected variable with a uniform random number
        p.position[r1] = r.generate_uniform_random_number(p.lb[r1], p.ub[r1])

        # Checks its limits
        p.clip_by_bound()

        # Evaluates its position
        p.fit = function(p.position)

        # If the new potision is better than agent's `j` position
        if p.fit < agent_j.fit:
            # Replaces the agent's `j` position and fitness
            agent_j.position = copy.deepcopy(p.position)
            agent_j.fit = copy.deepcopy(p.fit)

    def update(self, space: Space, function: Function) -> None:
        """Wraps Symbiotic Organisms Search over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Generates a random integer for mutualism and performs it
            j = r.generate_integer_random_number(0, len(space.agents), exclude_value=i)
            self._mutualism(agent, space.agents[j], space.best_agent, function)

            # Re-generates a random integer for commensalism and performs it
            j = r.generate_integer_random_number(0, len(space.agents), exclude_value=i)
            self._commensalism(agent, space.agents[j], space.best_agent, function)

            # Re-generates a random integer for parasitism and performs it
            j = r.generate_integer_random_number(0, len(space.agents), exclude_value=i)
            self._parasitism(agent, space.agents[j], function)

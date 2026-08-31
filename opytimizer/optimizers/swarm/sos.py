"""Symbiotic Organisms Search."""

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

import opytimizer.math.random as r
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


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

        super(SOS, self).__init__()

        self.build(params)

    def _mutualism(
        self, agent_i: Agent, agent_j: Agent, best_agent: Agent, function: Callable
    ) -> None:
        """Performs the mutualism operation.

        Args:
            agent_i: Selected `i` agent.
            agent_j: Selected `j` agent.
            best_agent: Global best agent.
            function: A callable that will be used as the objective function.

        """

        a = copy.deepcopy(agent_i)
        b = copy.deepcopy(agent_j)

        # Calculates the mutual vector (eq. 3)
        mutual_vector = (agent_i.position + agent_j.position) / 2

        BF_1, BF_2 = np.random.choice([1, 2], 2, replace=False)

        # Re-calculates the new positions (eq. 1 and 2)
        r1 = np.random.uniform(0.0, 1.0, 1)
        a.position += r1 * (best_agent.position - mutual_vector * BF_1)
        b.position += r1 * (best_agent.position - mutual_vector * BF_2)

        a.clip_by_bound()
        b.clip_by_bound()

        a.fit = function(a.position)
        b.fit = function(b.position)

        if a.fit < agent_i.fit:
            agent_i.position = copy.deepcopy(a.position)
            agent_i.fit = copy.deepcopy(a.fit)

        if b.fit < agent_j.fit:
            agent_j.position = copy.deepcopy(b.position)
            agent_j.fit = copy.deepcopy(b.fit)

    def _commensalism(
        self, agent_i: Agent, agent_j: Agent, best_agent: Agent, function: Callable
    ) -> None:
        """Performs the commensalism operation.

        Args:
            agent_i: Selected `i` agent.
            agent_j: Selected `j` agent.
            best_agent: Global best agent.
            function: A callable that will be used as the objective function.

        """

        a = copy.deepcopy(agent_i)

        # Updates the agent's position (eq. 4)
        r1 = np.random.uniform(-1, 1, 1)
        a.position += r1 * (best_agent.position - agent_j.position)
        a.clip_by_bound()

        a.fit = function(a.position)
        if a.fit < agent_i.fit:
            agent_i.position = copy.deepcopy(a.position)
            agent_i.fit = copy.deepcopy(a.fit)

    def _parasitism(self, agent_i: Agent, agent_j: Agent, function: Callable) -> None:
        """Performs the parasitism operation.

        Args:
            agent_i: Selected `i` agent.
            agent_j: Selected `j` agent.
            function: A callable that will be used as the objective function.

        """

        r1 = np.random.randint(0, agent_i.n_variables, None)

        p = copy.deepcopy(agent_i)
        p.position[r1] = np.random.uniform(p.lb[r1], p.ub[r1], 1)
        p.clip_by_bound()

        p.fit = function(p.position)
        if p.fit < agent_j.fit:
            agent_j.position = copy.deepcopy(p.position)
            agent_j.fit = copy.deepcopy(p.fit)

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Symbiotic Organisms Search over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        for i, agent in enumerate(space.agents):
            j = r.integer(0, len(space.agents), exclude=i, size=None)
            self._mutualism(agent, space.agents[j], space.best_agent, function)

            j = r.integer(0, len(space.agents), exclude=i, size=None)
            self._commensalism(agent, space.agents[j], space.best_agent, function)

            j = r.integer(0, len(space.agents), exclude=i, size=None)
            self._parasitism(agent, space.agents[j], function)

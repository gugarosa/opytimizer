"""Backtracking Search Optimization Algorithm."""

import copy
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import opytimizer.math.random as r
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class BSA(Optimizer):
    """A BSA class, inherited from Optimizer.

    This is the designed class to define BSOA-related
    variables and methods.

    References:
        P. Civicioglu. Backtracking search optimization algorithm for numerical optimization problems.
        Applied Mathematics and Computation (2013).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(BSA, self).__init__()

        self.F = 3.0
        self.mix_rate = 1

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.old_agents = copy.deepcopy(space.agents)

    def _permute(self, agents: List[Agent]) -> None:
        """Performs the permuting operator.

        Args:
            agents: List of agents.

        """

        a = np.random.uniform(0.0, 1.0, 1)
        b = np.random.uniform(0.0, 1.0, 1)

        if a < b:
            self.old_agents = copy.deepcopy(agents)

        i = np.random.randint(0, len(agents), None)
        j = r.integer(0, len(agents), exclude=i, size=None)

        self.old_agents[i], self.old_agents[j] = copy.deepcopy(
            self.old_agents[j]
        ), copy.deepcopy(self.old_agents[i])

    def _mutate(self, agents: List[Agent]) -> List[Agent]:
        """Performs the mutation operator.

        Args:
            agents: List of agents.

        Returns:
            (List[Agent]): A list holding the trial agents.

        """

        trial_agents = copy.deepcopy(agents)

        r1 = np.random.uniform(0.0, 1.0, 1)

        for trial_agent, agent, old_agent in zip(trial_agents, agents, self.old_agents):
            trial_agent.position = agent.position + self.F * r1 * (
                old_agent.position - agent.position
            )
            trial_agent.clip_by_bound()

        return trial_agents

    def _crossover(self, agents: List[Agent], trial_agents: List[Agent]) -> None:
        """Performs the crossover operator.

        Args:
            agents: List of agents.
            trial_agents: List of trial agents.

        """

        n_agents = len(agents)
        n_variables = agents[0].n_variables

        cross_map = np.ones((n_agents, n_variables))

        a = np.random.uniform(0.0, 1.0, 1)
        b = np.random.uniform(0.0, 1.0, 1)

        if a < b:
            for i in range(n_agents):
                r1 = np.random.uniform(0.0, 1.0, 1)

                non_crosses = int(self.mix_rate * r1 * n_variables)

                for _ in range(non_crosses):
                    u = np.random.randint(0, n_variables, None)
                    cross_map[i][u] = 0
        else:
            for i in range(n_agents):
                j = np.random.randint(0, n_variables, None)
                cross_map[i][j] = 0

        for i in range(n_agents):
            for j in range(n_variables):
                if cross_map[i][j]:
                    trial_agents[i].position[j] = copy.deepcopy(agents[i].position[j])

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Backtracking Search Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        self._permute(space.agents)
        trial_agents = self._mutate(space.agents)
        self._crossover(space.agents, trial_agents)

        for agent, trial_agent in zip(space.agents, trial_agents):
            trial_agent.fit = function(trial_agent.position)
            if trial_agent.fit < agent.fit:
                agent.position = copy.deepcopy(trial_agent.position)
                agent.fit = copy.deepcopy(trial_agent.fit)

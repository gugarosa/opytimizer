"""Germinal Center Optimization."""

import copy
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class GCO(Optimizer):
    """A GCO class, inherited from Optimizer.

    This is the designed class to define GCO-related
    variables and methods.

    References:
        C. Villaseñor et al. Germinal center optimization algorithm.
        International Journal of Computational Intelligence Systems (2018).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(GCO, self).__init__()

        self.CR = 0.7
        self.F = 1.25

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.life = np.random.uniform(70, 70, space.n_agents)
        self.counter = np.ones(space.n_agents)

    def _mutate_cell(
        self, agent: Agent, alpha: Agent, beta: Agent, gamma: Agent
    ) -> Agent:
        """Mutates a new cell based on distinct cells (alg. 2).

        Args:
            agent: Current agent.
            alpha: 1st picked agent.
            beta: 2nd picked agent.
            gamma: 3rd picked agent.

        Returns:
            (Agent): A mutated cell.

        """

        a = copy.deepcopy(agent)

        for j in range(a.n_variables):
            r2 = np.random.uniform(0.0, 1.0, 1)
            if r2 < self.CR:
                a.position[j] = alpha.position[j] + self.F * (
                    beta.position[j] - gamma.position[j]
                )

        return a

    def _dark_zone(self, agents: List[Agent], function: Callable) -> None:
        """Performs the dark-zone update process (alg. 1).

        Args:
            agents: List of agents.
            function: A callable that will be used as the objective function.

        """

        for i, agent in enumerate(agents):
            r1 = np.random.uniform(0, 100, 1)
            if r1 < self.life[i]:
                self.counter[i] += 1
            else:
                self.counter[i] = 1

            C = np.random.choice(
                len(agents), 3, p=self.counter / np.sum(self.counter), replace=False
            )

            a = self._mutate_cell(agent, agents[C[0]], agents[C[1]], agents[C[2]])
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

                self.life[i] += 10

    def _light_zone(self, agents: List[Agent]) -> None:
        """Performs the light-zone update process (alg. 1).

        Args:
            agents: List of agents.

        """

        fits = [agent.fit for agent in agents]
        min_fit, max_fit = np.min(fits), np.max(fits)

        for i, agent in enumerate(agents):
            self.life[i] = 10
            life_fit = (agent.fit - max_fit) / (min_fit - max_fit + c.EPSILON)
            self.life[i] += 10 * life_fit

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Germinal Center Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        self._dark_zone(space.agents, function)
        self._light_zone(space.agents)

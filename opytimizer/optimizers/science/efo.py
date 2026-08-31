"""Electromagnetic Field Optimization."""

import copy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class EFO(Optimizer):
    """An EFO class, inherited from Optimizer.

    This is the designed class to define EFO-related
    variables and methods.

    References:
        H. Abedinpourshotorban et al.
        Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm.
        Swarm and Evolutionary Computation (2016).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(EFO, self).__init__()

        self.positive_field = 0.1
        self.negative_field = 0.5

        self.ps_ratio = 0.1
        self.r_ratio = 0.4
        self.phi = (1 + np.sqrt(5)) / 2

        self.RI = 0

        self.build(params)

    def _calculate_indexes(self, n_agents: int) -> Tuple[int, int, int]:
        """Calculates the indexes of positive, negative and neutral particles.

        Args:
            n_agents: Number of agents in the space.

        Returns:
            (Tuple[int, int, int]): Positive, negative and neutral particles' indexes.

        """

        positive_index = int(np.random.uniform(0, n_agents * self.positive_field, 1))

        negative_index = int(
            np.random.uniform(n_agents * (1 - self.negative_field), n_agents, 1)
        )

        neutral_index = int(
            np.random.uniform(
                n_agents * self.positive_field, n_agents * (1 - self.negative_field), 1
            )
        )

        return positive_index, negative_index, neutral_index

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Electromagnetic Field Optimization over all agents and variables (eq. 1-4).

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        space.agents.sort(key=lambda x: x.fit)
        n_agents = len(space.agents)

        agent = copy.deepcopy(space.agents[0])
        force = np.random.uniform(0.0, 1.0, 1)

        for j in range(agent.n_variables):
            pos, neg, neu = self._calculate_indexes(n_agents)

            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 < self.ps_ratio:
                agent.position[j] = space.agents[pos].position[j]
            else:
                agent.position[j] = (
                    space.agents[neg].position[j]
                    + self.phi
                    * force
                    * (space.agents[pos].position[j] - space.agents[neu].position[j])
                    - force
                    * (space.agents[neg].position[j] - space.agents[neu].position[j])
                )
        agent.clip_by_bound()

        r2 = np.random.uniform(0.0, 1.0, 1)
        if r2 < self.r_ratio:
            agent.position[self.RI] = np.random.uniform(
                agent.lb[self.RI], agent.ub[self.RI], 1
            )

            self.RI += 1
            if self.RI >= agent.n_variables:
                self.RI = 1

        agent.fit = function(agent.position)
        if agent.fit < space.agents[-1].fit:
            space.agents[-1] = copy.deepcopy(agent)

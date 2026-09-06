# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Non-Dominated Sorting.

Compilation initializes domination counts, pairwise domination sets, and unknown statuses
of -1. Updates rank successive frontiers and count first-frontier points. Domination compares
agent position vectors using maximization: no objective is smaller and at least one is larger.

References:
    P. Godfrey, R. Shipley and J. Gryz.
    Algorithms and Analyses for Maximal Vector Computation. The VLDB Journal (2007).

"""

import copy
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class NDS(Optimizer):
    """Rank a population by successive non-dominated frontiers.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the Pareto point counter.

        Args:
            params: Overrides for the supported optimizer parameters.

        Notes:
            The supported key is ``n_pareto_points`` (initial first-frontier point count, 0).
            Updates accumulate this counter rather than resetting it.

        """

        super(NDS, self).__init__()

        self.n_pareto_points = 0

        self.build(params)

    def compile(self, space: Space) -> None:
        self.count = np.zeros(space.n_agents)
        self.set = np.zeros((space.n_agents, space.n_agents))

        self.status = np.full(space.n_agents, -1)

    def _compare_domination(self, agent_i: Agent, agent_j: Agent) -> bool:
        gt, gte = 0, 0

        n_objectives = agent_i.position.shape[0]
        for k in range(n_objectives):
            if agent_i.position[k] >= agent_j.position[k]:
                gte += 1

                if agent_i.position[k] > agent_j.position[k]:
                    gt += 1

        return gte == n_objectives and gt > 0

    def update(self, space: Space) -> None:
        temp_agents = copy.deepcopy(space.agents)
        temp_status = -10

        for i, agent in enumerate(space.agents):
            for j, temp in enumerate(temp_agents):
                if self._compare_domination(temp, agent):
                    self.count[i] += 1
                    self.set[j][i] = 1

        archive = []
        for i, agent in enumerate(space.agents):
            if self.count[i] == 0:
                self.status[i] = temp_status
                archive.append(i)

                self.n_pareto_points += 1

        aux_archive = []
        while len(archive) != 0:
            temp_status -= 1

            for f in archive:
                for s in self.set[f].nonzero()[0]:
                    self.count[s] -= 1

                    if self.count[s] == 0:
                        aux_archive.append(s)
                        self.status[s] = temp_status

            archive = aux_archive
            aux_archive = []

        for i, agent in enumerate(space.agents):
            old_status = self.status[i]

            if old_status != -1:
                self.status[i] = old_status - temp_status - 1

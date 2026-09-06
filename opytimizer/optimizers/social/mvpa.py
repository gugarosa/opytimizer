# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Most Valuable Player Algorithm.

References:
    H. Bouchekara. Most Valuable Player Algorithm: a novel optimization algorithm inspired from sport.
    Operational Research (2017).

"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class MVPA(Optimizer):
    """Implement Most Valuable Player Algorithm.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Configure competing teams.

        Args:
            params: Attribute overrides applied without copying their values.

        Notes:
            ``n_teams`` (4) is the number of teams.
            Compilation derives players per team, with any remainder in the final team.

        """

        super(MVPA, self).__init__()

        self.n_teams = 4

        self.build(params)

    def compile(self, space: Space) -> None:
        self.n_p = space.n_agents // self.n_teams

    def _get_agents_from_team(self, agents: list[Agent], index: int) -> list[Agent]:
        start, end = index * self.n_p, (index + 1) * self.n_p

        if (index + 1) == self.n_teams:
            return sorted(agents[start:], key=lambda x: x.fit)

        return sorted(agents[start:end], key=lambda x: x.fit)

    def update(self, space: Space, function: Callable) -> None:
        for i in range(self.n_teams):
            team_i = self._get_agents_from_team(space.agents, i)
            franchise_i = copy.deepcopy(team_i[0])
            fitness_i = np.mean([agent.fit for agent in team_i])

            j = r.integer(0, self.n_teams, exclude=i, size=None)
            team_j = self._get_agents_from_team(space.agents, j)
            franchise_j = copy.deepcopy(team_j[0])
            fitness_j = np.mean([agent.fit for agent in team_j])

            for agent in team_i:
                a = copy.deepcopy(agent)

                r1 = np.random.uniform(0.0, 1.0, 1)
                r2 = np.random.uniform(0.0, 1.0, 1)
                r3 = np.random.uniform(0.0, 1.0, 1)

                # Updates temporary agent's position (eq. 9)
                a.position += r1 * (franchise_i.position - a.position) + 2 * r1 * (
                    space.best_agent.position - a.position
                )

                # Calculates the probability of team `i` beating team `j` (eq. 16)
                Pr = 1 - fitness_i / (fitness_i + fitness_j + c.EPSILON)

                if r2 < Pr:
                    # Updates temporary agent's position (eq. 17)
                    a.position += r3 * (a.position - franchise_j.position)
                else:
                    # Updates temporary agent's position (eq. 18)
                    a.position += r3 * (franchise_j.position - a.position)
                a.clip_by_bound()

                a.fit = function(a.position)
                if a.fit < agent.fit:
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

"""Most Valuable Player Algorithm.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class MVPA(Optimizer):
    """A MVPA class, inherited from Optimizer.

    This is the designed class to define MVPA-related
    variables and methods.

    References:
        H. Bouchekara. Most Valuable Player Algorithm: a novel optimization algorithm inspired from sport.
        Operational Research (2017).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> MVPA.")

        # Overrides its parent class with the receiving params
        super(MVPA, self).__init__()

        # Maximum number of teams
        self.n_teams = 4

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def n_teams(self) -> int:
        """Maximum number of teams."""

        return self._n_teams

    @n_teams.setter
    def n_teams(self, n_teams: int) -> None:
        if not isinstance(n_teams, int):
            raise e.TypeError("`n_teams` should be an integer")
        if n_teams < 1:
            raise e.ValueError("`n_teams` should be > 0")

        self._n_teams = n_teams

    @property
    def n_p(self) -> int:
        """Number of players per team."""

        return self._n_p

    @n_p.setter
    def n_p(self, n_p: int) -> None:
        if not isinstance(n_p, int):
            raise e.TypeError("`n_p` should be an integer")
        if n_p < 1:
            raise e.ValueError("`n_p` should be > 0")

        self._n_p = n_p

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Number of players per team
        self.n_p = space.n_agents // self.n_teams

    def _get_agents_from_team(self, agents: List[Agent], index: int) -> List[Agent]:
        """Gets a set of agents from a specified team.

        Args:
            agents: List of agents.
            index: Index of team.

        Returns:
            (List[Agent]): A sorted list of agents that belongs to the specified team.

        """

        # Defines the starting and ending points
        start, end = index * self.n_p, (index + 1) * self.n_p

        # If it is the last index, there is no need to return an ending point
        if (index + 1) == self.n_teams:
            return sorted(agents[start:], key=lambda x: x.fit)

        return sorted(agents[start:end], key=lambda x: x.fit)

    def update(self, space: Space, function: Function) -> None:
        """Wraps Most Valuable Player Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Iterates through every team
        for i in range(self.n_teams):
            # Gets the agents for the specified team, its franchise agent and the mean fitness
            team_i = self._get_agents_from_team(space.agents, i)
            franchise_i = copy.deepcopy(team_i[0])
            fitness_i = np.mean([agent.fit for agent in team_i])

            # Gets the agents for the opposite team, its franchise agent and the mean fitness
            j = r.generate_integer_random_number(0, self.n_teams, i)
            team_j = self._get_agents_from_team(space.agents, j)
            franchise_j = copy.deepcopy(team_j[0])
            fitness_j = np.mean([agent.fit for agent in team_j])

            # Iterates through all agents in current team
            for agent in team_i:
                # Makes a deep copy of current agent
                a = copy.deepcopy(agent)

                # Generates uniform random numbers
                r1 = r.generate_uniform_random_number()
                r2 = r.generate_uniform_random_number()
                r3 = r.generate_uniform_random_number()

                # Updates temporary agent's position (eq. 9)
                a.position += r1 * (franchise_i.position - a.position) + 2 * r1 * (
                    space.best_agent.position - a.position
                )

                # Calculates the probability of team `i` beating team `j` (eq. 16)
                Pr = 1 - fitness_i / (fitness_i + fitness_j + c.EPSILON)

                # If random number is smaller than probability
                if r2 < Pr:
                    # Updates temporary agent's position (eq. 17)
                    a.position += r3 * (a.position - franchise_j.position)

                # If random number is bigger than probability
                else:
                    # Updates temporary agent's position (eq. 18)
                    a.position += r3 * (franchise_j.position - a.position)

                # Clips the temporary agent
                a.clip_by_bound()

                # Re-evaluates its fitness
                a.fit = function(a.position)

                # If temporary agent is better than old one
                if a.fit < agent.fit:
                    # Copies its position and fitness to the agent
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

"""Most Valuable Player Algorithm.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core import Optimizer

logger = l.get_logger(__name__)


class MVPA(Optimizer):
    """A MVPA class, inherited from Optimizer.

    This is the designed class to define MVPA-related
    variables and methods.

    References:
        H. Bouchekara. Most Valuable Player Algorithm: a novel optimization algorithm inspired from sport.
        Operational Research (2017).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> MVPA.')

        # Overrides its parent class with the receiving params
        super(MVPA, self).__init__()

        # Maximum number of teams
        self.n_teams = 4

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    @property
    def n_teams(self):
        """int: Maximum number of teams.

        """

        return self._n_teams

    @n_teams.setter
    def n_teams(self, n_teams):
        if not isinstance(n_teams, int):
            raise e.TypeError('`n_teams` should be an integer')
        if n_teams < 1:
            raise e.ValueError('`n_teams` should be > 0')

        self._n_teams = n_teams

    @property
    def n_p(self):
        """int: Number of players per team.

        """

        return self._n_p

    @n_p.setter
    def n_p(self, n_p):
        if not isinstance(n_p, int):
            raise e.TypeError('`n_p` should be an integer')
        if n_p < 1:
            raise e.ValueError('`n_p` should be > 0')

        self._n_p = n_p

    def compile(self, space):
        """Compiles additional information that is used by this optimizer.

        Args:
            space (Space): A Space object containing meta-information.

        """

        # Number of players per team
        self.n_p = space.n_agents // self.n_teams

    def _get_agents_from_team(self, agents, index):
        """Gets a set of agents from a specified team.

        Args:
            agents (list): List of agents.
            index (int): Index of team.

        Returns:
            A sorted list of agents that belongs to the specified team.

        """

        # Defines the starting and ending points
        start, end = index * self.n_p, (index + 1) * self.n_p

        # If it is the last index, there is no need to return an ending point
        if (index + 1) == self.n_teams:
            return sorted(agents[start:], key=lambda x: x.fit)

        return sorted(agents[start:end], key=lambda x: x.fit)

    def update(self, space, function):
        """Wraps Most Valuable Player Algorithm over all agents and variables.

        Args:
            space (Space): Space containing agents and update-related information.
            function (Function): A Function object that will be used as the objective function.

        """

        # Iterates through every team
        for i in range(self.n_teams):
            # Gets the agents for the specified team and its franchise agent
            team_agents = self._get_agents_from_team(space.agents, i)
            franchise_agent = copy.deepcopy(team_agents[0])

            # Gets the opposite team
            j = r.generate_integer_random_number(0, self.n_teams, i)
            opp_agents = self._get_agents_from_team(space.agents, j)

            # Iterates through all agents in team
            for agent in team_agents:
                # Updates the agent's position (eq. 9)
                r1 = r.generate_uniform_random_number()
                agent.position += r1 * (franchise_agent.position - agent.position) + 2 * r1 * (space.best_agent.position - agent.position)

                #
                r2 = r.generate_uniform_random_number()

                if r2 < 0.5:
                    r3 = r.generate_uniform_random_number()
                    agent.position += r3 * (agent.position - opp_agents[0].position)
                else:
                    r3 = r.generate_uniform_random_number()
                    agent.position += r3 * (opp_agents[0].position - agent.position)

                #
                agent.clip_by_bound()
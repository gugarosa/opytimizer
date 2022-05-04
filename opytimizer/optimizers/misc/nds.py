"""Non-Dominated Sorting.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class NDS(Optimizer):
    """An NDS class, inherited from Optimizer.

    This is the designed class to define NDS-related
    variables and methods.

    References:
        P. Godfrey, R. Shipley and J. Gryz.
        Algorithms and Analyses for Maximal Vector Computation.
        The VLDB Journal (2007).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        # Overrides its parent class with the receiving params
        super(NDS, self).__init__()

        # Number of points in the frontier
        self.n_pareto_points = 0

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def n_pareto_points(self) -> int:
        """Number of points in the frontier."""

        return self._n_pareto_points

    @n_pareto_points.setter
    def n_pareto_points(self, n_pareto_points: int) -> None:
        if not isinstance(n_pareto_points, int):
            raise e.TypeError("`n_pareto_points` should be an integer")
        if n_pareto_points < 0:
            raise e.ValueError("`n_pareto_points` should be >= 0")

        self._n_pareto_points = n_pareto_points

    @property
    def count(self) -> np.ndarray:
        """Array of domination counts."""

        return self._count

    @count.setter
    def count(self, count: np.ndarray) -> None:
        if not isinstance(count, np.ndarray):
            raise e.TypeError("`count` should be a numpy array")

        self._count = count

    @property
    def set(self) -> np.ndarray:
        """Array of dominating set."""

        return self._set

    @set.setter
    def set(self, set: np.ndarray) -> None:
        if not isinstance(set, np.ndarray):
            raise e.TypeError("`set` should be a numpy array")

        self._set = set

    @property
    def status(self) -> np.ndarray:
        """Array of pareto status."""

        return self._status

    @status.setter
    def status(self, status: np.ndarray) -> None:
        if not isinstance(status, np.ndarray):
            raise e.TypeError("`status` should be a numpy array")

        self._status = status

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Array of domination counts
        self.count = np.zeros(space.n_agents)

        # Array of dominating sets
        self.set = np.zeros((space.n_agents, space.n_agents))

        # Array of pareto status
        # -1 = unknown, 0 = pareto, 1 = non-pareto
        self.status = np.full(space.n_agents, -1)

    def _compare_domination(self, agent_i: Agent, agent_j: Agent) -> bool:
        """Calculates whether `i` dominates `j`.

        Args:
            agent_i: Agent `i`.
            agent_j: Agent `j`.

        Returns:
            (bool): Boolean indicating whether `i` dominated `j` or not.

        """

        # Counts of greater than and greater/equal than
        # between `i` and `j`
        gt, gte = 0, 0

        # Gathers the number of objectives
        n_objectives = agent_i.position.shape[0]

        for k in range(n_objectives):
            # Compares if `i` is greater/equal than `j`
            if agent_i.position[k] >= agent_j.position[k]:
                gte += 1

                # Compares if `i` is greater than `j`
                if agent_i.position[k] > agent_j.position[k]:
                    gt += 1

        return gte == n_objectives and gt > 0

    def update(self, space: Space) -> None:
        """Wraps Non-Dominated Sorting over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        # Copies a temporary list for iterating purposes
        # and defines a temporary status
        temp_agents = copy.deepcopy(space.agents)
        temp_status = -10

        # Iterates through 'i' agents
        for i, agent in enumerate(space.agents):
            # Iterates through 'j' agents
            for j, temp in enumerate(temp_agents):
                # Performs a domination comparison between `i` and `j`
                if self._compare_domination(temp, agent):
                    # Increments the counter
                    self.count[i] += 1

                    # And adds the dominated solution to the set
                    self.set[j][i] = 1

        # Finds the first archive (frontier)
        archive = []
        for i, agent in enumerate(space.agents):
            # If the solution is non-dominant, it should be
            # added to the frontier
            if self.count[i] == 0:
                self.status[i] = temp_status
                archive.append(i)

                # Increments the number of points in the frontier
                self.n_pareto_points += 1

        # Finds the subsequence archives (frontiers)
        aux_archive = []
        while len(archive) != 0:
            temp_status -= 1

            # Iterates through every solution in current frontier
            for f in archive:
                # Checks solutions that are dominated by current solution
                for s in self.set[f].nonzero()[0]:
                    self.count[s] -= 1

                    # Adds to the auxiliary frontier if solution is non-dominant
                    if self.count[s] == 0:
                        aux_archive.append(s)
                        self.status[s] = temp_status

            # When current frontier is explored, resets the auxiliary frontier
            archive = aux_archive
            aux_archive = []

        # Adjusts the rankings based on the found frontiers
        for i, agent in enumerate(space.agents):
            old_status = self.status[i]

            if old_status != -1:
                self.status[i] = old_status - temp_status - 1

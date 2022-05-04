"""Brain Storm Optimization.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class BSO(Optimizer):
    """A BSO class, inherited from Optimizer.

    This is the designed class to define BSO-related
    variables and methods.

    References:
        Y. Shi. Brain Storm Optimization Algorithm.
        International Conference in Swarm Intelligence (2011).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> BSO.")

        # Overrides its parent class with the receiving params
        super(BSO, self).__init__()

        # Number of clusters
        self.m = 5

        # Probability of replacing a random cluster
        self.p_replacement_cluster = 0.2

        # Probability of selecting a single cluster
        self.p_single_cluster = 0.8

        # Probability of selecting the best idea from a single cluster
        self.p_single_best = 0.4

        # Probability of selecting the best idea from a pair of clusters
        self.p_double_best = 0.5

        # Controls the sigmoid's slope
        self.k = 20

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def m(self) -> int:
        """Number of clusters."""

        return self._m

    @m.setter
    def m(self, m: int) -> None:
        if not isinstance(m, int):
            raise e.TypeError("`m` should be an integer")
        if m <= 0:
            raise e.ValueError("`m` should be > 0")

        self._m = m

    @property
    def p_replacement_cluster(self) -> float:
        """Probability of replacing a random cluster."""

        return self._p_replacement_cluster

    @p_replacement_cluster.setter
    def p_replacement_cluster(self, p_replacement_cluster: float) -> None:
        if not isinstance(p_replacement_cluster, (float, int)):
            raise e.TypeError("`p_replacement_cluster` should be a float or integer")
        if p_replacement_cluster < 0 or p_replacement_cluster > 1:
            raise e.ValueError("`p_replacement_cluster` should be between 0 and 1")

        self._p_replacement_cluster = p_replacement_cluster

    @property
    def p_single_cluster(self) -> float:
        """Probability of selecting a single cluster."""

        return self._p_single_cluster

    @p_single_cluster.setter
    def p_single_cluster(self, p_single_cluster: float) -> None:
        if not isinstance(p_single_cluster, (float, int)):
            raise e.TypeError("`p_single_cluster` should be a float or integer")
        if p_single_cluster < 0 or p_single_cluster > 1:
            raise e.ValueError("`p_single_cluster` should be between 0 and 1")

        self._p_single_cluster = p_single_cluster

    @property
    def p_single_best(self) -> float:
        """Probability of selecting the best idea from a single cluster."""

        return self._p_single_best

    @p_single_best.setter
    def p_single_best(self, p_single_best: float) -> None:
        if not isinstance(p_single_best, (float, int)):
            raise e.TypeError("`p_single_best` should be a float or integer")
        if p_single_best < 0 or p_single_best > 1:
            raise e.ValueError("`p_single_best` should be between 0 and 1")

        self._p_single_best = p_single_best

    @property
    def p_double_best(self) -> float:
        """Probability of selecting the best idea from a pair of clusters."""

        return self._p_double_best

    @p_double_best.setter
    def p_double_best(self, p_double_best: float) -> None:
        if not isinstance(p_double_best, (float, int)):
            raise e.TypeError("`p_double_best` should be a float or integer")
        if p_double_best < 0 or p_double_best > 1:
            raise e.ValueError("`p_double_best` should be between 0 and 1")

        self._p_double_best = p_double_best

    @property
    def k(self) -> float:
        """Controls the sigmoid's slope."""

        return self._k

    @k.setter
    def k(self, k: float) -> None:
        if not isinstance(k, (float, int)):
            raise e.TypeError("`k` should be a float or integer")
        if k <= 0:
            raise e.ValueError("`k` should should be > 0")

        self._k = k

    def _clusterize(self, agents: List[Agent]) -> Tuple[np.ndarray, np.ndarray]:
        """Performs the clusterization over the agents' positions.

        Args:
            agents: List of agents.

        Returns:
            (Tuple[np.ndarray, np.ndarray]): Agents indexes and best agent index per cluster.

        """

        # Gathers current agents' positions (ideas)
        ideas = np.array([agent.position for agent in agents])

        # Performs the K-means clustering
        labels = g.kmeans(ideas, self.m)

        # Creates lists to ideas and best idea indexes per cluster
        ideas_idx_per_cluster, best_idx_per_cluster = [], []

        # Iterates through all possible clusters
        for i in range(self.m):
            # Gathers ideas that belongs to current cluster
            ideas_idx = np.where(labels == i)[0]

            # If there are any ideas
            if len(ideas_idx) > 0:
                # Appends them to the corresponding list
                ideas_idx_per_cluster.append(ideas_idx)

            # If not
            else:
                # Just appends an empty list for compatibility purposes
                ideas_idx_per_cluster.append([])

            # Gathers a tuple of sorted agents and their index for the current cluster
            ideas_per_cluster = [(agents[j], j) for j in ideas_idx_per_cluster[i]]
            ideas_per_cluster.sort(key=lambda x: x[0].fit)

            # If there are any ideas
            if len(ideas_per_cluster) > 0:
                # Appends the best index to the corresponding list
                best_idx_per_cluster.append(ideas_per_cluster[0][1])

            # If not
            else:
                # Just appends a `-1` for compatibility purposes
                best_idx_per_cluster.append(-1)

        return ideas_idx_per_cluster, best_idx_per_cluster

    def _sigmoid(self, x: float) -> float:
        """Calculates the sigmoid function.

        Args:
            x: Input value.

        Returns:
            Output value.

        """

        return 1 / (1 + np.exp(-x))

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Brain Storm Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Number of iterations.s

        """

        # Clusterizes the current agents
        ideas_idx_per_cluster, best_idx_per_cluster = self._clusterize(space.agents)

        # Generates a random number
        r1 = r.generate_uniform_random_number()

        # If random number is smaller than probability of replacement
        if r1 < self.p_replacement_cluster:
            # Selects a random cluster
            c = r.generate_integer_random_number(0, self.m)

            # Fills agent with a new uniform position
            space.agents[best_idx_per_cluster[c]].fill_with_uniform()

        # Iterates through all agents
        for agent in space.agents:
            # Makes a deep copy of current agent
            a = copy.deepcopy(agent)

            # Generates a random number
            r2 = r.generate_uniform_random_number()

            # If random number is smaller than probability of selecting a single cluster
            if r2 < self.p_single_cluster:
                # Randomly selects a cluster
                c = r.generate_integer_random_number(0, self.m)

                # If there are available ideas in the cluster
                if len(ideas_idx_per_cluster[c]) > 0:
                    # Generates a random number
                    r3 = r.generate_uniform_random_number()

                    # If selection should come from best cluster
                    if r3 < self.p_single_best:
                        # Updates the temporary agent's position
                        a.position = copy.deepcopy(
                            space.agents[best_idx_per_cluster[c]].position
                        )

                    # If selection should come from a random agent in cluster
                    else:
                        # Gathers an index from random agent in cluster
                        j = r.generate_integer_random_number(
                            0, len(ideas_idx_per_cluster[c])
                        )

                        # Updates the temporary agent's position
                        a.position = copy.deepcopy(
                            space.agents[ideas_idx_per_cluster[c][j]].position
                        )

            # If random number is bigger than probability of selecting a single cluster
            else:
                # Checks if there are two or more available clusters
                if self.m > 1:
                    # Selects two different clusters
                    c1 = r.generate_integer_random_number(0, self.m)
                    c2 = r.generate_integer_random_number(0, self.m, c1)

                    # If both clusters have at least one idea
                    if (
                        len(ideas_idx_per_cluster[c1]) > 0
                        and len(ideas_idx_per_cluster[c2]) > 0
                    ):
                        # Generates a new set of random numbers
                        r4 = r.generate_uniform_random_number()

                        # If selection should come from best clusters
                        if r4 < self.p_double_best:
                            # Updates the temporary agent's position
                            a.position = (
                                space.agents[best_idx_per_cluster[c1]].position
                                + space.agents[best_idx_per_cluster[c2]].position
                            ) / 2

                        # If selection should come from random agents in clusters
                        else:
                            # Gathers indexes from agents in clusters
                            u = r.generate_integer_random_number(
                                0, len(ideas_idx_per_cluster[c1])
                            )
                            v = r.generate_integer_random_number(
                                0, len(ideas_idx_per_cluster[c2])
                            )

                            # Updates the temporary agent's position
                            a.position = (
                                space.agents[ideas_idx_per_cluster[c1][u]].position
                                + space.agents[ideas_idx_per_cluster[c2][v]].position
                            ) / 2

            # Generates a random noise and activates it with a sigmoid function
            r5 = r.generate_uniform_random_number()
            csi = self._sigmoid((0.5 * n_iterations - iteration) / self.k) * r5

            # Updates the temporary agent's position
            a.position += csi * r.generate_gaussian_random_number()

            # Checks agent's limits
            a.clip_by_bound()

            # Re-evaluates the temporary agent
            a.fit = function(a.position)

            # If temporary agent's fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Replace its position and fitness
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

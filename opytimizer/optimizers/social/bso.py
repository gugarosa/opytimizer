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

        super(BSO, self).__init__()

        self.m = 5

        self.p_replacement_cluster = 0.2
        self.p_single_cluster = 0.8
        self.p_single_best = 0.4
        self.p_double_best = 0.5

        self.k = 20

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

        ideas = np.array([agent.position for agent in agents])
        labels = g.kmeans(ideas, self.m)

        ideas_idx_per_cluster, best_idx_per_cluster = [], []

        for i in range(self.m):
            ideas_idx = np.where(labels == i)[0]

            if len(ideas_idx) > 0:
                ideas_idx_per_cluster.append(ideas_idx)
            else:
                ideas_idx_per_cluster.append([])

            ideas_per_cluster = [(agents[j], j) for j in ideas_idx_per_cluster[i]]
            ideas_per_cluster.sort(key=lambda x: x[0].fit)

            if len(ideas_per_cluster) > 0:
                best_idx_per_cluster.append(ideas_per_cluster[0][1])
            else:
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

        ideas_idx_per_cluster, best_idx_per_cluster = self._clusterize(space.agents)

        r1 = r.generate_uniform_random_number()
        if r1 < self.p_replacement_cluster:
            c = r.generate_integer_random_number(0, self.m)
            space.agents[best_idx_per_cluster[c]].fill_with_uniform()

        for agent in space.agents:
            a = copy.deepcopy(agent)

            r2 = r.generate_uniform_random_number()
            if r2 < self.p_single_cluster:
                c = r.generate_integer_random_number(0, self.m)
                if len(ideas_idx_per_cluster[c]) > 0:
                    r3 = r.generate_uniform_random_number()
                    if r3 < self.p_single_best:
                        a.position = copy.deepcopy(
                            space.agents[best_idx_per_cluster[c]].position
                        )
                    else:
                        j = r.generate_integer_random_number(
                            0, len(ideas_idx_per_cluster[c])
                        )

                        a.position = copy.deepcopy(
                            space.agents[ideas_idx_per_cluster[c][j]].position
                        )
            else:
                if self.m > 1:
                    c1 = r.generate_integer_random_number(0, self.m)
                    c2 = r.generate_integer_random_number(0, self.m, c1)

                    if (
                        len(ideas_idx_per_cluster[c1]) > 0
                        and len(ideas_idx_per_cluster[c2]) > 0
                    ):
                        r4 = r.generate_uniform_random_number()
                        if r4 < self.p_double_best:
                            a.position = (
                                space.agents[best_idx_per_cluster[c1]].position
                                + space.agents[best_idx_per_cluster[c2]].position
                            ) / 2
                        else:
                            u = r.generate_integer_random_number(
                                0, len(ideas_idx_per_cluster[c1])
                            )
                            v = r.generate_integer_random_number(
                                0, len(ideas_idx_per_cluster[c2])
                            )

                            a.position = (
                                space.agents[ideas_idx_per_cluster[c1][u]].position
                                + space.agents[ideas_idx_per_cluster[c2][v]].position
                            ) / 2

            r5 = r.generate_uniform_random_number()
            csi = self._sigmoid((0.5 * n_iterations - iteration) / self.k) * r5

            a.position += csi * r.generate_gaussian_random_number()
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

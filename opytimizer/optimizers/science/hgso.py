"""Henry Gas Solubility Optimization.
"""

from typing import Any, Dict, Optional

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


class HGSO(Optimizer):
    """An HGSO class, inherited from Optimizer.

    This is the designed class to define HGSO-related
    variables and methods.

    References:
        F. Hashim et al. Henry gas solubility optimization: A novel physics-based algorithm.
        Future Generation Computer Systems (2019).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> HGSO.")

        # Overrides its parent class with the receiving params
        super(HGSO, self).__init__()

        # Number of clusters
        self.n_clusters = 2

        # Henry's coefficient constant
        self.l1 = 0.0005

        # Partial pressure constant
        self.l2 = 100

        # Constant
        self.l3 = 0.001

        # Influence of gases
        self.alpha = 1.0

        # Gas constant
        self.beta = 1.0

        # Solubility constant
        self.K = 1.0

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""

        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, n_clusters: int) -> None:
        if not isinstance(n_clusters, int):
            raise e.TypeError("`n_clusters` should be an integer")
        if n_clusters <= 0:
            raise e.ValueError("`n_clusters` should be > 0")

        self._n_clusters = n_clusters

    @property
    def l1(self) -> float:
        """Henry's coefficient constant."""

        return self._l1

    @l1.setter
    def l1(self, l1: float) -> None:
        if not isinstance(l1, (float, int)):
            raise e.TypeError("`l1` should be a float or integer")
        if l1 < 0:
            raise e.ValueError("`l1` should be >= 0")

        self._l1 = l1

    @property
    def l2(self) -> int:
        """Partial pressure constant."""

        return self._l2

    @l2.setter
    def l2(self, l2: int) -> None:
        if not isinstance(l2, int):
            raise e.TypeError("`l2` should be an integer")
        if l2 <= 0:
            raise e.ValueError("`l2` should be > 0")

        self._l2 = l2

    @property
    def l3(self) -> float:
        """Constant."""

        return self._l3

    @l3.setter
    def l3(self, l3: float) -> None:
        if not isinstance(l3, (float, int)):
            raise e.TypeError("`l3` should be a float or integer")
        if l3 < 0:
            raise e.ValueError("`l3` should be >= 0")

        self._l3 = l3

    @property
    def alpha(self) -> float:
        """Influence of gases."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        if alpha < 0:
            raise e.ValueError("`alpha` should be >= 0")

        self._alpha = alpha

    @property
    def beta(self) -> float:
        """Gas constant."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        if beta < 0:
            raise e.ValueError("`beta` should be >= 0")

        self._beta = beta

    @property
    def K(self) -> float:
        """Solubility constant."""

        return self._K

    @K.setter
    def K(self, K: float) -> None:
        if not isinstance(K, (float, int)):
            raise e.TypeError("`K` should be a float or integer")
        if K < 0:
            raise e.ValueError("`K` should be >= 0")

        self._K = K

    @property
    def coefficient(self) -> np.ndarray:
        """Array of coefficients."""

        return self._coefficient

    @coefficient.setter
    def coefficient(self, coefficient: np.ndarray) -> None:
        if not isinstance(coefficient, np.ndarray):
            raise e.TypeError("`coefficient` should be a numpy array")

        self._coefficient = coefficient

    @property
    def pressure(self) -> np.ndarray:
        """Array of pressures."""

        return self._pressure

    @pressure.setter
    def pressure(self, pressure: np.ndarray) -> None:
        if not isinstance(pressure, np.ndarray):
            raise e.TypeError("`pressure` should be a numpy array")

        self._pressure = pressure

    @property
    def constant(self) -> np.ndarray:
        """Array of constants."""

        return self._constant

    @constant.setter
    def constant(self, constant: np.ndarray) -> None:
        if not isinstance(constant, np.ndarray):
            raise e.TypeError("`constant` should be a numpy array")

        self._constant = constant

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Number of agents per cluster
        n_agents_per_cluster = int(len(space.agents) / self.n_clusters)

        # Arrays of coefficients, pressures and constants
        self.coefficient = self.l1 * r.generate_uniform_random_number(
            size=self.n_clusters
        )
        self.pressure = self.l2 * r.generate_uniform_random_number(
            size=(self.n_clusters, n_agents_per_cluster)
        )
        self.constant = self.l3 * r.generate_uniform_random_number(size=self.n_clusters)

    def _update_position(
        self, agent: Agent, cluster_agent: Agent, best_agent: Agent, solubility: float
    ) -> np.ndarray:
        """Updates the position of a single gas (eq. 10).

        Args:
            agent: Current agent.
            cluster_agent: Best cluster's agent.
            best_agent: Best agent.
            solubility: Solubility for current agent.

        Returns:
            (np.ndarray): An updated position.

        """

        # Calculates `gamma`
        gamma = self.beta * np.exp(-(best_agent.fit + 0.05) / (agent.fit + 0.05))

        # Generates a flag that provides diversity
        flag = np.sign(r.generate_uniform_random_number(-1, 1))

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number()

        # Updates the position
        new_position = (
            agent.position
            + flag * r1 * gamma * (cluster_agent.position - agent.position)
            + flag
            * r1
            * self.alpha
            * (solubility * best_agent.position - agent.position)
        )

        return new_position

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Henry Gas Solubility Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Creates n-wise clusters
        clusters = g.n_wise(space.agents, self.pressure.shape[1])

        # Iterates through all clusters
        for i, cluster in enumerate(clusters):
            # Calculates the system's current temperature (eq. 8)
            T = np.exp(-iteration / n_iterations)

            # Updates Henry's coefficient (eq. 8)
            self.coefficient[i] *= np.exp(-self.constant[i] * (1 / T - 1 / 298.15))

            # Transforms the cluster into a list and sorts it
            cluster = list(cluster)
            cluster.sort(key=lambda x: x.fit)

            # Iterates through all agents in cluster
            for j, agent in enumerate(cluster):
                # Calculates agent's solubility (eq. 9)
                solubility = self.K * self.coefficient[i] * self.pressure[i][j]

                # Updates agent's position (eq. 10)
                agent.position = self._update_position(
                    agent, cluster[0], space.best_agent, solubility
                )

                # Clips agent's limits
                agent.clip_by_bound()

                # Re-calculates its fitness
                agent.fit = function(agent.position)

        # Re-sorts the whole space
        space.agents.sort(key=lambda x: x.fit)

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number()

        # Calculates the number of worst agents (eq. 11)
        N = int(len(space.agents) * (r1 * (0.2 - 0.1) + 0.1))

        # Iterates through every bad agent
        for agent in space.agents[-N:]:
            # Generates another uniform random number
            r2 = r.generate_uniform_random_number()

            # Updates bad agent's position (eq. 12)
            agent.position = agent.lb + r2 * (agent.ub - agent.lb)

"""Henry Gas Solubility Optimization."""

from typing import Any, Callable, Dict, Optional

import numpy as np

import opytimizer.math.general as g
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


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

        super(HGSO, self).__init__()

        self.n_clusters = 2

        self.l1 = 0.0005
        self.l2 = 100
        self.l3 = 0.001

        self.alpha = 1.0
        self.beta = 1.0
        self.K = 1.0

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        n_agents_per_cluster = int(len(space.agents) / self.n_clusters)

        self.coefficient = self.l1 * np.random.uniform(0.0, 1.0, self.n_clusters)
        self.pressure = self.l2 * np.random.uniform(
            0.0, 1.0, (self.n_clusters, n_agents_per_cluster)
        )
        self.constant = self.l3 * np.random.uniform(0.0, 1.0, self.n_clusters)

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

        gamma = self.beta * np.exp(-(best_agent.fit + 0.05) / (agent.fit + 0.05))
        flag = np.sign(np.random.uniform(-1, 1, 1))

        r1 = np.random.uniform(0.0, 1.0, 1)

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
        self, space: Space, function: Callable, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Henry Gas Solubility Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        clusters = g.n_wise(space.agents, self.pressure.shape[1])
        for i, cluster in enumerate(clusters):
            # Calculates the system's current temperature (eq. 8)
            T = np.exp(-iteration / n_iterations)

            # Updates Henry's coefficient (eq. 8)
            self.coefficient[i] *= np.exp(-self.constant[i] * (1 / T - 1 / 298.15))

            cluster = list(cluster)
            cluster.sort(key=lambda x: x.fit)

            for j, agent in enumerate(cluster):
                # Calculates agent's solubility (eq. 9)
                solubility = self.K * self.coefficient[i] * self.pressure[i][j]

                # Updates agent's position (eq. 10)
                agent.position = self._update_position(
                    agent, cluster[0], space.best_agent, solubility
                )
                agent.clip_by_bound()

                agent.fit = function(agent.position)

        space.agents.sort(key=lambda x: x.fit)

        # Calculates the number of worst agents (eq. 11)
        r1 = np.random.uniform(0.0, 1.0, 1)
        N = int(len(space.agents) * (r1 * (0.2 - 0.1) + 0.1))

        for agent in space.agents[-N:]:
            # Updates bad agent's position (eq. 12)
            r2 = np.random.uniform(0.0, 1.0, 1)
            agent.position = agent.lb + r2 * (agent.ub - agent.lb)

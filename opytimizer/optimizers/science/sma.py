"""Slime Mould Algorithm.
"""

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


class SMA(Optimizer):
    """A SMA class, inherited from Optimizer.

    This is the designed class to define SMA-related
    variables and methods.

    References:
        S. Li, H. Chen, M. Wang, A. A. Heidari, S. Mirjalili
        Slime mould algorithm: A new method for stochastic optimization.
        Future Generation Computer Systems (2020).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> SMA.")

        super(SMA, self).__init__()

        self.z = 0.03

        self.build(params)

        logger.info("Class overrided.")

    @property
    def z(self) -> float:
        """Probability threshold."""

        return self._z

    @z.setter
    def z(self, z: float) -> None:
        if not isinstance(z, (float, int)):
            raise e.TypeError("`z` should be a float or integer")
        if z < 0:
            raise e.ValueError("`z` should be >= 0")

        self._z = z

    @property
    def weight(self) -> np.ndarray:
        """Array of weights."""

        return self._weight

    @weight.setter
    def weight(self, weight: np.ndarray) -> None:
        if not isinstance(weight, np.ndarray):
            raise e.TypeError("`weight` should be a numpy array")

        self._weight = weight

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.
        Args:
            space: A Space object containing meta-information.
        """

        self.weight = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

    def _update_weight(self, agents: List[Agent]):
        """Updates the weight of slime mould (eq. 2.5).

        Args:
            agents: List of agents.

        """

        best, worst = agents[0].fit, agents[-1].fit

        n_agents = len(agents)

        for i in range(n_agents):

            r1 = r.generate_uniform_random_number(
                0, 1, (agents[i].n_variables, agents[i].n_dimensions)
            )

            if i <= int(n_agents / 2):
                self.weight[i] = 1 + r1 * np.log10(
                    (best - agents[i].fit) / ((best - worst) + c.EPSILON) + 1
                )
            else:
                self.weight[i] = 1 - r1 * np.log10(
                    (best - agents[i].fit) / ((best - worst) + c.EPSILON) + 1
                )

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        """Wraps Slime Mould Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A function object.

        """

        space.agents.sort(key=lambda x: x.fit)

        self._update_weight(space.agents)

        a = np.arctanh(-((iteration + 1) / (n_iterations + 1)) + 1)
        b = 1 - (iteration + 1) / (n_iterations + 1)

        for i, agent in enumerate(space.agents):

            r2 = r.generate_uniform_random_number()

            if r2 < self.z:
                agent.fill_with_uniform()
            else:
                p = np.tanh(np.abs(agent.fit - space.agents[0].fit))
                vb = r.generate_uniform_random_number(-a, a)
                vc = r.generate_uniform_random_number(-b, b)

                r3 = r.generate_uniform_random_number()

                if r3 < p:
                    k = r.generate_integer_random_number(0, len(space.agents))
                    l = r.generate_integer_random_number(
                        0, len(space.agents), exclude_value=k
                    )
                    agent.position = space.agents[0].position + vb * (
                        self.weight[i]
                        * (space.agents[k].position - space.agents[l].position)
                    )
                else:
                    agent.position *= vc

"""Flower Pollination Algorithm.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class FPA(Optimizer):
    """A FPA class, inherited from Optimizer.

    This is the designed class to define FPA-related
    variables and methods.

    References:
        X.-S. Yang. Flower pollination algorithm for global optimization.
        International conference on unconventional computing and natural computation (2012).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(FPA, self).__init__()

        self.beta = 1.5
        self.eta = 0.2
        self.p = 0.8

        self.build(params)

        logger.info("Class overrided.")

    @property
    def beta(self) -> float:
        """Lévy flight control parameter."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        if beta <= 0 or beta > 2:
            raise e.ValueError("`beta` should be between 0 and 2")

        self._beta = beta

    @property
    def eta(self) -> float:
        """Lévy flight scaling factor."""

        return self._eta

    @eta.setter
    def eta(self, eta: float) -> None:
        if not isinstance(eta, (float, int)):
            raise e.TypeError("`eta` should be a float or integer")
        if eta < 0:
            raise e.ValueError("`eta` should be >= 0")

        self._eta = eta

    @property
    def p(self) -> float:
        """Probability of local pollination."""

        return self._p

    @p.setter
    def p(self, p: float) -> None:
        if not isinstance(p, (float, int)):
            raise e.TypeError("`p` should be a float or integer")
        if p < 0 or p > 1:
            raise e.ValueError("`p` should be between 0 and 1")

        self._p = p

    def _global_pollination(
        self, agent_position: np.ndarray, best_position: np.ndarray
    ) -> np.ndarray:
        """Updates the agent's position based on a global pollination (eq. 1).

        Args:
            agent_position: Agent's current position.
            best_position: Best agent's current position.

        Returns:
            (np.ndarray): A new position.

        """

        step = d.generate_levy_distribution(self.beta)
        global_pollination = self.eta * step * (best_position - agent_position)
        new_position = agent_position + global_pollination

        return new_position

    def _local_pollination(
        self,
        agent_position: np.ndarray,
        k_position: np.ndarray,
        l_position: np.ndarray,
        epsilon: float,
    ) -> np.ndarray:
        """Updates the agent's position based on a local pollination (eq. 3).

        Args:
            agent_position: Agent's current position.
            k_position: Agent's (index k) current position.
            l_position: Agent's (index l) current position.
            epsilon: An uniform random generated number.

        Returns:
            (np.ndarray): A new position.

        """

        local_pollination = epsilon * (k_position - l_position)
        new_position = agent_position + local_pollination

        return new_position

    def update(self, space: Space, function: Function) -> None:
        """Wraps Flower Pollination Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        for agent in space.agents:
            a = copy.deepcopy(agent)

            r1 = r.generate_uniform_random_number()
            if r1 > self.p:
                a.position = self._global_pollination(
                    agent.position, space.best_agent.position
                )
            else:
                epsilon = r.generate_uniform_random_number()

                k = r.generate_integer_random_number(0, len(space.agents))
                l = r.generate_integer_random_number(
                    0, len(space.agents), exclude_value=k
                )

                a.position = self._local_pollination(
                    agent.position,
                    space.agents[k].position,
                    space.agents[l].position,
                    epsilon,
                )
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

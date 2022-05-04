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

        # Overrides its parent class with the receiving params
        super(FPA, self).__init__()

        # Lévy flight control parameter
        self.beta = 1.5

        # Lévy flight scaling factor
        self.eta = 0.2

        # Probability of local pollination
        self.p = 0.8

        # Builds the class
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

        # Generates a Lévy distribution
        step = d.generate_levy_distribution(self.beta)

        # Calculates the global pollination
        global_pollination = self.eta * step * (best_position - agent_position)

        # Calculates the new position based on previous global pollination
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

        # Calculates the local pollination
        local_pollination = epsilon * (k_position - l_position)

        # Calculates the new position based on previous local pollination
        new_position = agent_position + local_pollination

        return new_position

    def update(self, space: Space, function: Function) -> None:
        """Wraps Flower Pollination Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Iterates through all agents
        for agent in space.agents:
            # Creates a temporary agent
            a = copy.deepcopy(agent)

            # Generates an uniform random number
            r1 = r.generate_uniform_random_number()

            # Check if generated random number is bigger than probability
            if r1 > self.p:
                # Updates a temporary position according to global pollination
                a.position = self._global_pollination(
                    agent.position, space.best_agent.position
                )

            else:
                # Generates an uniform random number
                epsilon = r.generate_uniform_random_number()

                # Generates an index for flower `k` and flower `l`
                k = r.generate_integer_random_number(0, len(space.agents))
                l = r.generate_integer_random_number(
                    0, len(space.agents), exclude_value=k
                )

                # Updates a temporary position according to local pollination
                a.position = self._local_pollination(
                    agent.position,
                    space.agents[k].position,
                    space.agents[l].position,
                    epsilon,
                )

            # Checks agent's limits
            a.clip_by_bound()

            # Calculates the fitness for the temporary position
            a.fit = function(a.position)

            # If new fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Copies its position and fitness to the agent
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

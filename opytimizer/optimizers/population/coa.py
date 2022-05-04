"""Coyote Optimization Algorithm.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class COA(Optimizer):
    """A COA class, inherited from Optimizer.

    This is the designed class to define COA-related
    variables and methods.

    References:
        J. Pierezan and L. Coelho. Coyote Optimization Algorithm: A New Metaheuristic for Global Optimization Problems.
        IEEE Congress on Evolutionary Computation (2018).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> COA.")

        # Overrides its parent class with the receiving params
        super(COA, self).__init__()

        # Number of packs
        self.n_p = 2

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def n_p(self) -> int:
        """Number of packs."""

        return self._n_p

    @n_p.setter
    def n_p(self, n_p: int) -> None:
        if not isinstance(n_p, int):
            raise e.TypeError("`n_p` should be an integer")
        if n_p <= 0:
            raise e.ValueError("`n_p` should be > 0")

        self._n_p = n_p

    @property
    def n_c(self) -> int:
        """Number of coyotes per pack."""

        return self._n_c

    @n_c.setter
    def n_c(self, n_c: int) -> None:
        if not isinstance(n_c, int):
            raise e.TypeError("`n_c` should be an integer")
        if n_c <= 0:
            raise e.ValueError("`n_c` should be > 0")

        self._n_c = n_c

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Calculates the number of coyotes per pack
        self.n_c = space.n_agents // self.n_p

    def _get_agents_from_pack(self, agents: List[Agent], index: int) -> List[Agent]:
        """Gets a set of agents from a specified pack.

        Args:
            agents: List of agents.
            index: Index of pack.

        Returns:
            (List[Agent]): A sorted list of agents that belongs to the specified pack.

        """

        # Defines the starting and ending points
        start, end = index * self.n_c, (index + 1) * self.n_c

        # If it is the last index, there is no need to return an ending point
        if (index + 1) == self.n_p:
            return sorted(agents[start:], key=lambda x: x.fit)

        return sorted(agents[start:end], key=lambda x: x.fit)

    def _transition_packs(self, agents: List[Agent]) -> None:
        """Transits coyotes between packs (eq. 4).

        Args:
            agents: List of agents.

        """

        # Calculates the eviction probability
        p_e = 0.005 * len(agents)

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number()

        # If random number is smaller than eviction probability
        if r1 < p_e:
            # Gathers two random packs
            p1 = r.generate_integer_random_number(high=self.n_p)
            p2 = r.generate_integer_random_number(high=self.n_p)

            # Gathers two random coyotes
            c1 = r.generate_integer_random_number(high=self.n_c)
            c2 = r.generate_integer_random_number(high=self.n_c)

            # Calculates their indexes
            i = self.n_c * p1 + c1
            j = self.n_c * p2 + c2

            # Performs a swap between them
            agents[i], agents[j] = copy.deepcopy(agents[j]), copy.deepcopy(agents[i])

    def update(self, space: Space, function: Function) -> None:
        """Wraps Coyote Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Iterates through all packs
        for i in range(self.n_p):
            # Gets the agents for the specified pack
            pack_agents = self._get_agents_from_pack(space.agents, i)

            # Gathers the alpha coyote (eq. 5)
            alpha = pack_agents[0]

            # Computes the cultural tendency (eq. 6)
            tendency = np.median(
                np.array([agent.position for agent in pack_agents]), axis=0
            )

            # Iterates through all coyotes in the pack
            for agent in pack_agents:
                # Makes a deep copy of current coyote
                a = copy.deepcopy(agent)

                # Generates two random integers
                cr1 = r.generate_integer_random_number(high=len(pack_agents))
                cr2 = r.generate_integer_random_number(high=len(pack_agents))

                # Calculates the alpha and pack influences
                lambda_1 = alpha.position - pack_agents[cr1].position
                lambda_2 = tendency - pack_agents[cr2].position

                # Generates two random uniform numbers
                r1 = r.generate_uniform_random_number()
                r2 = r.generate_uniform_random_number()

                # Updates the social condition (eq. 12)
                a.position += r1 * lambda_1 + r2 * lambda_2

                # Checks the agent's limits
                a.clip_by_bound()

                # Evaluates the agent (eq. 13)
                a.fit = function(a.position)

                # If the new potision is better than current agent's position (eq. 14)
                if a.fit < agent.fit:
                    # Replaces the current agent's position and fitness
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

            # Performs transition between packs (eq. 4)
            self._transition_packs(space.agents)

"""Coyote Optimization Algorithm."""

import copy
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


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

        super(COA, self).__init__()

        self.n_p = 2

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.n_c = space.n_agents // self.n_p

    def _get_agents_from_pack(self, agents: List[Agent], index: int) -> List[Agent]:
        """Gets a set of agents from a specified pack.

        Args:
            agents: List of agents.
            index: Index of pack.

        Returns:
            (List[Agent]): A sorted list of agents that belongs to the specified pack.

        """

        start, end = index * self.n_c, (index + 1) * self.n_c

        if (index + 1) == self.n_p:
            return sorted(agents[start:], key=lambda x: x.fit)

        return sorted(agents[start:end], key=lambda x: x.fit)

    def _transition_packs(self, agents: List[Agent]) -> None:
        """Transits coyotes between packs (eq. 4).

        Args:
            agents: List of agents.

        """

        p_e = 0.005 * len(agents)
        r1 = np.random.uniform(0.0, 1.0, 1)

        if r1 < p_e:
            p1 = np.random.randint(0, self.n_p, None)
            p2 = np.random.randint(0, self.n_p, None)

            c1 = np.random.randint(0, self.n_c, None)
            c2 = np.random.randint(0, self.n_c, None)

            i = self.n_c * p1 + c1
            j = self.n_c * p2 + c2

            agents[i], agents[j] = copy.deepcopy(agents[j]), copy.deepcopy(agents[i])

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Coyote Optimization Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        for i in range(self.n_p):
            pack_agents = self._get_agents_from_pack(space.agents, i)

            # Gathers the alpha coyote (eq. 5)
            alpha = pack_agents[0]

            # Computes the cultural tendency (eq. 6)
            tendency = np.median(
                np.array([agent.position for agent in pack_agents]), axis=0
            )

            for agent in pack_agents:
                a = copy.deepcopy(agent)

                cr1 = np.random.randint(0, len(pack_agents), None)
                cr2 = np.random.randint(0, len(pack_agents), None)

                lambda_1 = alpha.position - pack_agents[cr1].position
                lambda_2 = tendency - pack_agents[cr2].position

                r1 = np.random.uniform(0.0, 1.0, 1)
                r2 = np.random.uniform(0.0, 1.0, 1)

                # Updates the social condition (eq. 12)
                a.position += r1 * lambda_1 + r2 * lambda_2
                a.clip_by_bound()

                # Evaluates the agent (eq. 13)
                a.fit = function(a.position)

                # If the new potision is better than current agent's position (eq. 14)
                if a.fit < agent.fit:
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

            # Performs transition between packs (eq. 4)
            self._transition_packs(space.agents)

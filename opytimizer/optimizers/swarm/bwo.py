"""Black Widow Optimization."""

import copy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

import opytimizer.math.random as r
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class BWO(Optimizer):
    """A BWO class, inherited from Optimizer.

    This is the designed class to define BWO-related
    variables and methods.

    References:
        V. Hayyolalam and A. Kazem.
        Black Widow Optimization Algorithm: A novel meta-heuristic approach
        for solving engineering optimization problems.
        Engineering Applications of Artificial Intelligence (2020).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(BWO, self).__init__()

        self.pp = 0.6
        self.cr = 0.44
        self.pm = 0.4

        self.build(params)

    def _procreating(self, x1: Agent, x2: Agent) -> Tuple[Agent, Agent]:
        """Procreates a pair of parents into offsprings (eq. 1).

        Args:
            x1: Father to produce the offsprings.
            x2: Mother to produce the offsprings.

        Returns:
            (Tuple[Agent, Agent]): Two generated offsprings based on parents.

        """

        y1, y2 = copy.deepcopy(x1), copy.deepcopy(x2)

        alpha = np.random.uniform(0.0, 1.0, 1)
        y1.position = alpha * x1.position + (1 - alpha) * x2.position
        y2.position = alpha * x2.position + (1 - alpha) * x1.position

        return y1, y2

    def _mutation(self, alpha: Agent) -> Agent:
        """Performs the mutation over an offspring (s. 3.4).

        Args:
            alpha: Offspring to be mutated.

        Returns:
            (Agent): The mutated offspring.

        """

        if alpha.n_variables > 1:
            r1 = np.random.randint(0, alpha.n_variables, None)
            r2 = r.integer(0, alpha.n_variables, exclude=r1, size=None)

            alpha.position[r1], alpha.position[r2] = (
                alpha.position[r2],
                alpha.position[r1],
            )

        return alpha

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Black Widow Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        n_agents = len(space.agents)
        n_variables = space.n_variables

        n_reproduct = int(n_agents * self.pp)
        n_cannibals = int(n_agents * self.cr)
        n_mutate = int(n_agents * self.pm)

        space.agents.sort(key=lambda x: x.fit)

        agents1 = copy.deepcopy(space.agents[:n_reproduct])
        agents2 = []

        for _ in range(0, n_reproduct):
            idx = np.random.uniform(0, n_agents, 2)

            father, mother = copy.deepcopy(space.agents[int(idx[0])]), copy.deepcopy(
                space.agents[int(idx[1])]
            )

            new_agents = []

            for _ in range(0, int(n_variables / 2)):
                y1, y2 = self._procreating(father, mother)

                y1.clip_by_bound()
                y2.clip_by_bound()

                y1.fit = function(y1.position)
                y2.fit = function(y2.position)

                new_agents.extend([mother, y1, y2])

            new_agents.sort(key=lambda x: x.fit)

            # Extending auxiliary population with the number of cannibals (s. 3.3)
            agents2.extend(new_agents[:n_cannibals])

        for _ in range(0, n_mutate):
            idx = int(np.random.uniform(0, n_reproduct))

            alpha = self._mutation(agents1[idx])
            alpha.clip_by_bound()

            alpha.fit = function(alpha.position)

            agents2.extend([alpha])

        space.agents += agents2
        space.agents.sort(key=lambda x: x.fit)
        space.agents = space.agents[:n_agents]

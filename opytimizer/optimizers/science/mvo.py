"""Multi-Verse Optimizer."""

from typing import Any, Callable, Dict, Optional

import numpy as np

import opytimizer.math.general as g
from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class MVO(Optimizer):
    """A MVO class, inherited from Optimizer.

    This is the designed class to define MVO-related
    variables and methods.

    References:
        S. Mirjalili, S. M. Mirjalili and A. Hatamlou.
        Multi-verse optimizer: a nature-inspired algorithm for global optimization.
        Neural Computing and Applications (2016).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(MVO, self).__init__()

        self.WEP_min = 0.2
        self.WEP_max = 1.0

        self.p = 6.0

        self.build(params)

    def update(
        self, space: Space, function: Callable, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Multi-Verse Optimizer over all agents and variables (eq. 3.1-3.4).

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        WEP = self.WEP_min + (iteration + 1) * (
            (self.WEP_max - self.WEP_min) / n_iterations
        )
        TDR = 1 - ((iteration + 1) ** (1 / self.p) / n_iterations ** (1 / self.p))

        fitness = [agent.fit for agent in space.agents]

        norm = np.linalg.norm(fitness)
        norm_fitness = fitness / norm

        for i, agent in enumerate(space.agents):
            for j in range(agent.n_variables):
                r1 = np.random.uniform(0.0, 1.0, 1)
                if r1 < norm_fitness[i]:
                    white_hole = g.weighted_wheel_selection(norm_fitness)
                    agent.position[j] = space.agents[white_hole].position[j]

                r2 = np.random.uniform(0.0, 1.0, 1)
                if r2 < WEP:
                    width = np.random.uniform(agent.lb[j], agent.ub[j], 1)

                    r3 = np.random.uniform(0.0, 1.0, 1)
                    if r3 < 0.5:
                        agent.position[j] = space.best_agent.position[j] + TDR * width
                    else:
                        agent.position[j] = space.best_agent.position[j] - TDR * width
            agent.clip_by_bound()

            agent.fit = function(agent.position)

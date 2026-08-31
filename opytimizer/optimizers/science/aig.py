"""Algorithm of the Innovative Gunner."""

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class AIG(Optimizer):
    """An AIG class, inherited from Optimizer.

    This is the designed class to define AIG-related
    variables and methods.

    References:
        P. Pijarski and P. Kacejko.
        A new metaheuristic optimization method: the algorithm of the innovative gunner (AIG).
        Engineering Optimization (2019).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(AIG, self).__init__()

        self.alpha = np.pi
        self.beta = np.pi

        self.build(params)

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Algorithm of the Innovative Gunner over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        # Calculates the maximum correction angles (eq. 18)
        a = np.random.uniform(0.0, 1.0, 1)
        alpha_max = self.alpha * a
        beta_max = self.beta * a

        for agent in space.agents:
            a = copy.deepcopy(agent)

            alpha = np.random.normal(
                0, alpha_max / 3, (agent.n_variables, agent.n_dimensions)
            )
            beta = np.random.normal(
                0, beta_max / 3, (agent.n_variables, agent.n_dimensions)
            )

            # Calculates correction functions (eq. 16 and 17)
            g_alpha = np.where(alpha < 0, np.cos(alpha), 1 / np.cos(alpha))
            g_beta = np.where(beta < 0, np.cos(beta), 1 / np.cos(beta))

            # Updates temporary agent's position (eq. 15)
            a.position *= g_alpha * g_beta
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

"""Simulated Annealing."""

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class SA(Optimizer):
    """A SA class, inherited from Optimizer.

    This is the designed class to define SA-related
    variables and methods.

    References:
        A. Khachaturyan, S. Semenovsovskaya and B. Vainshtein.
        The thermodynamic approach to the structure analysis of crystals.
        Acta Crystallographica (1981).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(SA, self).__init__()

        self.T = 100
        self.beta = 0.999

        self.build(params)

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Simulated Annealing over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A function object.

        """

        for agent in space.agents:
            a = copy.deepcopy(agent)

            noise = np.random.normal(0, 0.1, (agent.n_variables, agent.n_dimensions))

            a.position += noise
            a.clip_by_bound()

            r1 = np.random.uniform(0.0, 1.0, 1)
            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)
            elif r1 < np.exp(-(a.fit - agent.fit) / self.T):
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

        self.T *= self.beta

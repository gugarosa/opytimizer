"""Electro-Search Algorithm."""

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class ESA(Optimizer):
    """An ESA class, inherited from Optimizer.

    This is the designed class to define ES-related
    variables and methods.

    References:
        A. Tabari and A. Ahmad. A new optimization method: Electro-Search algorithm.
        Computers & Chemical Engineering (2017).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(ESA, self).__init__()

        self.n_electrons = 5

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.D = np.random.uniform(
            0.0, 1.0, (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def update(self, space: Space, function: Callable) -> None:
        """Wraps EElectro-Search Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        for i, agent in enumerate(space.agents):
            a = copy.deepcopy(agent)

            electrons = [copy.deepcopy(agent) for _ in range(self.n_electrons)]
            for electron in electrons:
                r1 = np.random.uniform(0.0, 1.0, 1)
                n = np.random.randint(2, 6, None)

                # Updates the electron's position (eq. 3)
                electron.position += (2 * r1 - 1) * (1 - 1 / n**2) / self.D[i]
                electron.clip_by_bound()

                electron.fit = function(electron.position)

            electrons.sort(key=lambda x: x.fit)

            # Generates both Rydberg constant and acceleration coefficient
            # Original implementation is missing up an informative description
            Re = np.random.uniform(0.0, 1.0, 1)
            Ac = np.random.uniform(0.0, 1.0, 1)

            # Updates the Orbital radius (eq. 4)
            self.D[i] = (electrons[0].position - space.best_agent.position) + Re * (
                1 / space.best_agent.position**2 - 1 / a.position**2
            )

            # Updates the temporary agent's position (eq. 5)
            a.position += Ac * self.D[i]
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

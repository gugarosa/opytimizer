"""Crow Search Algorithm."""

import copy
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.space import Space


class CSA(Optimizer):
    """A CSA class, inherited from Optimizer.

    This is the designed class to define CSA-related
    variables and methods.

    References:
        A. Askarzadeh. A novel metaheuristic method for
        solving constrained engineering optimization problems: Crow search algorithm.
        Computers & Structures (2016).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(CSA, self).__init__()

        self.fl = 2.0
        self.AP = 0.1

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.memory = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

    def evaluate(self, space: Space, function: Callable) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A Space object that will be evaluated.
            function: A callable that will be used as the objective function.

        """

        for i, agent in enumerate(space.agents):
            fit = function(agent.position)
            if fit < agent.fit:
                agent.fit = fit

                # Updates the memory to current's agent position (eq. 5)
                self.memory[i] = copy.deepcopy(agent.position)

            if agent.fit < space.best_agent.fit:
                space.best_agent.position = copy.deepcopy(self.memory[i])
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: Space) -> None:
        """Wraps Crow Search Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        for agent in space.agents:
            r1 = np.random.uniform(0.0, 1.0, 1)
            r2 = np.random.uniform(0.0, 1.0, 1)

            # Generates a random integer (e.g. selects the crow)
            j = np.random.randint(0, len(space.agents), None)

            if r1 >= self.AP:
                # Updates agent's position (eq. 2)
                agent.position += r2 * self.fl * (self.memory[j] - agent.position)
            else:
                agent.fill_with_uniform()

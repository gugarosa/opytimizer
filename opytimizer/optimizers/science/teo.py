"""Thermal Exchange Optimization."""

import copy
from typing import Any, Dict, Optional

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class TEO(Optimizer):
    """A TEO class, inherited from Optimizer.

    This is the designed class to define TEO-related
    variables and methods.

    References:
        A. Kaveh and A. Dadras. A novel meta-heuristic optimization algorithm: Thermal exchange optimization.
        Advances in Engineering Software (2017).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(TEO, self).__init__()

        self.c1 = True
        self.c2 = True

        self.pro = 0.05

        self.n_TM = 4
        self.TM = []

        self.build(params)

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.environment = copy.deepcopy(space.agents)

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        """Wraps Thermal Exchange Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        space.agents.sort(key=lambda x: x.fit)

        self.TM.append(copy.deepcopy(space.agents[0]))
        self.TM = self.TM[-self.n_TM :]

        space.agents = space.agents[: -len(self.TM)] + self.TM
        space.agents.sort(key=lambda x: x.fit)

        # Calculates the time (eq. 9)
        time = iteration / n_iterations

        for env in self.environment:
            # Updates the environment's position (eq. 10)
            r1 = np.random.uniform(0.0, 1.0, 1)
            env.position = 1 - (self.c1 + self.c2 * (1 - time)) * r1 * env.position

        for agent, env in zip(space.agents, self.environment):
            # Calculates the agent's beta value (eq. 8)
            beta = agent.fit / space.agents[-1].fit

            # Updates the agent's position (eq. 11)
            agent.position = env.position + (agent.position - env.position) * np.exp(
                -beta * time
            )

            r1 = np.random.uniform(0.0, 1.0, 1)
            if r1 < self.pro:
                idx = np.random.randint(0, agent.n_variables, None)

                # Resets its position (eq. 12)
                r2 = np.random.uniform(0.0, 1.0, 1)
                agent.position[idx] = agent.lb[idx] + r2 * (
                    agent.ub[idx] - agent.lb[idx]
                )

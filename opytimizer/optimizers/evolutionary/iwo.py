"""Invasive Weed Optimization."""

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class IWO(Optimizer):
    """An IWO class, inherited from Optimizer.

    This is the designed class to define IWO-related
    variables and methods.

    References:
        A. R. Mehrabian and C. Lucas. A novel numerical optimization algorithm inspired from weed colonization.
        Ecological informatics (2006).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(IWO, self).__init__()

        self.min_seeds = 0
        self.max_seeds = 5

        self.e = 2

        self.final_sigma = 0.001
        self.init_sigma = 3.0
        self.sigma = 0.0

        self.build(params)

    def _spatial_dispersal(self, iteration: int, n_iterations: int) -> None:
        """Calculates the Spatial Dispersal coefficient (eq. 1).

        Args:
            iteration: Current iteration number.
            n_iterations: Maximum number of iterations.

        """

        coef = ((n_iterations - iteration) ** self.e) / (
            (n_iterations + c.EPSILON) ** self.e
        )

        self.sigma = coef * (self.init_sigma - self.final_sigma) + self.final_sigma

    def _produce_offspring(self, agent: Agent, function: Callable) -> Agent:
        """Reproduces and flowers a seed into a new offpsring.

        Args:
            agent: An agent instance to be reproduced.
            function: A callable that will be used as the objective function.

        Returns:
            (Agent): An evolved offspring.

        """

        a = copy.deepcopy(agent)

        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            a.position[j] += self.sigma * np.random.uniform(lb, ub, a.n_dimensions)
        a.clip_by_bound()

        a.fit = function(a.position)

        return a

    def update(
        self, space: Space, function: Callable, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Invasive Weed Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        self._spatial_dispersal(iteration, n_iterations)

        n_agents = len(space.agents)
        offsprings = []

        space.agents.sort(key=lambda x: x.fit)

        for agent in space.agents:
            ratio = (agent.fit - space.agents[-1].fit) / (
                space.agents[0].fit - space.agents[-1].fit + c.EPSILON
            )

            n_seeds = int(self.min_seeds + (self.max_seeds - self.min_seeds) * ratio)
            for _ in range(n_seeds):
                a = self._produce_offspring(agent, function)
                offsprings.append(a)

        space.agents += offsprings
        space.agents.sort(key=lambda x: x.fit)
        space.agents = space.agents[:n_agents]

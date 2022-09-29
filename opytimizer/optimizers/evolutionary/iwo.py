"""Invasive Weed Optimization.
"""

import copy
from typing import Any, Dict, Optional

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as ex
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


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

        logger.info("Class overrided.")

    @property
    def min_seeds(self) -> int:
        """Minimum number of seeds."""

        return self._min_seeds

    @min_seeds.setter
    def min_seeds(self, min_seeds: int) -> None:
        if not isinstance(min_seeds, int):
            raise ex.TypeError("`min_seeds` should be an integer")
        if min_seeds < 0:
            raise ex.ValueError("`min_seeds` should be >= 0")

        self._min_seeds = min_seeds

    @property
    def max_seeds(self) -> int:
        """Maximum number of seeds."""

        return self._max_seeds

    @max_seeds.setter
    def max_seeds(self, max_seeds: int) -> None:
        if not isinstance(max_seeds, int):
            raise ex.TypeError("`max_seeds` should be an integer")
        if max_seeds < self.min_seeds:
            raise ex.ValueError("`max_seeds` should be >= `min_seeds`")

        self._max_seeds = max_seeds

    @property
    def e(self) -> float:
        """Exponent used to calculate the Spatial Dispersal."""

        return self._e

    @e.setter
    def e(self, e: float) -> None:
        if not isinstance(e, (float, int)):
            raise ex.TypeError("`e` should be a float or integer")
        if e < 0:
            raise ex.ValueError("`e` should be >= 0")

        self._e = e

    @property
    def final_sigma(self) -> float:
        """Final standard deviation."""

        return self._final_sigma

    @final_sigma.setter
    def final_sigma(self, final_sigma: float) -> None:
        if not isinstance(final_sigma, (float, int)):
            raise ex.TypeError("`final_sigma` should be a float or integer")
        if final_sigma < 0:
            raise ex.ValueError("`final_sigma` should be >= 0")

        self._final_sigma = final_sigma

    @property
    def init_sigma(self) -> float:
        """Initial standard deviation."""

        return self._init_sigma

    @init_sigma.setter
    def init_sigma(self, init_sigma: float) -> None:
        if not isinstance(init_sigma, (float, int)):
            raise ex.TypeError("`init_sigma` should be a float or integer")
        if init_sigma < 0:
            raise ex.ValueError("`init_sigma` should be >= 0")
        if init_sigma < self.final_sigma:
            raise ex.ValueError("`init_sigma` should be >= `final_sigma`")

        self._init_sigma = init_sigma

    @property
    def sigma(self) -> float:
        """Standard deviation."""

        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float) -> None:
        if not isinstance(sigma, (float, int)):
            raise ex.TypeError("`sigma` should be a float or integer")

        self._sigma = sigma

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

    def _produce_offspring(self, agent: Agent, function: Function) -> Agent:
        """Reproduces and flowers a seed into a new offpsring.

        Args:
            agent: An agent instance to be reproduced.
            function: A Function object that will be used as the objective function.

        Returns:
            (Agent): An evolved offspring.

        """

        a = copy.deepcopy(agent)

        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            a.position[j] += self.sigma * r.generate_uniform_random_number(
                lb, ub, a.n_dimensions
            )
        a.clip_by_bound()

        a.fit = function(a.position)

        return a

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Invasive Weed Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
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

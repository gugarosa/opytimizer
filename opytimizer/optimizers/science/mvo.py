"""Multi-Verse Optimizer.
"""

from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


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

        logger.info("Class overrided.")

    @property
    def WEP_min(self) -> float:
        """Minimum Wormhole Existence Probability."""

        return self._WEP_min

    @WEP_min.setter
    def WEP_min(self, WEP_min: float) -> None:
        if not isinstance(WEP_min, (float, int)):
            raise e.TypeError("`WEP_min` should be a float or integer")
        if WEP_min < 0 or WEP_min > 1:
            raise e.ValueError("`WEP_min` should be >= 0 and < 1")

        self._WEP_min = WEP_min

    @property
    def WEP_max(self) -> float:
        """Maximum Wormhole Existence Probability."""

        return self._WEP_max

    @WEP_max.setter
    def WEP_max(self, WEP_max: float) -> None:
        if not isinstance(WEP_max, (float, int)):
            raise e.TypeError("`WEP_max` should be a float or integer")
        if WEP_max < 0 or WEP_max > 1:
            raise e.ValueError("`WEP_max` should be >= 0 and < 1")
        if WEP_max < self.WEP_min:
            raise e.ValueError("`WEP_max` should be >= `WEP_min`")

        self._WEP_max = WEP_max

    @property
    def p(self) -> float:
        """Exploitation accuracy."""

        return self._p

    @p.setter
    def p(self, p: float) -> None:
        if not isinstance(p, (float, int)):
            raise e.TypeError("`p` should be a float or integer")
        if p < 0:
            raise e.ValueError("`p` should be >= 0")

        self._p = p

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Multi-Verse Optimizer over all agents and variables (eq. 3.1-3.4).

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
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
                r1 = r.generate_uniform_random_number()
                if r1 < norm_fitness[i]:
                    white_hole = g.weighted_wheel_selection(norm_fitness)
                    agent.position[j] = space.agents[white_hole].position[j]

                r2 = r.generate_uniform_random_number()
                if r2 < WEP:
                    width = r.generate_uniform_random_number(agent.lb[j], agent.ub[j])

                    r3 = r.generate_uniform_random_number()
                    if r3 < 0.5:
                        agent.position[j] = space.best_agent.position[j] + TDR * width
                    else:
                        agent.position[j] = space.best_agent.position[j] - TDR * width
            agent.clip_by_bound()

            agent.fit = function(agent.position)

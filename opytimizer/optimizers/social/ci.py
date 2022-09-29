"""Cohort Intelligence.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.general as g
import opytimizer.math.random as rnd
import opytimizer.utils.exception as e
from opytimizer.core.function import Function
from opytimizer.core.optimizer import Optimizer
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class CI(Optimizer):
    """A CI class, inherited from Optimizer.

    This is the designed class to define CI-related
    variables and methods.

    References:
        A. J. Kulkarni, I. P. Durugkar, M. Kumar. Cohort Intelligence: A Self Supervised Learning Behavior.
        IEEE International Conference on Systems, Man, and Cybernetics (2013).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> CI.")

        super(CI, self).__init__()

        self.r = 0.8
        self.t = 3

        self.build(params)

        logger.info("Class overrided.")

    @property
    def r(self) -> float:
        """Sampling interval reduction factor."""

        return self._r

    @r.setter
    def r(self, r: float) -> None:
        if not isinstance(r, (float, int)):
            raise e.TypeError("`r` should be a float or integer")
        if r < 0 or r > 1:
            raise e.ValueError("`r` should be between 0 and 1")

        self._r = r

    @property
    def t(self) -> int:
        """Number of variations."""

        return self._t

    @t.setter
    def t(self, t: int) -> None:
        if not isinstance(t, int):
            raise e.TypeError("`t` should be an integer")
        if t <= 0:
            raise e.ValueError("`t` should be > 0")

        self._t = t

    @property
    def lower(self) -> np.ndarray:
        """Array of lower bounds."""

        return self._lower

    @lower.setter
    def lower(self, lower: np.ndarray) -> None:
        if not isinstance(lower, np.ndarray):
            raise e.TypeError("`lower` should be a numpy array")

        self._lower = lower

    @property
    def upper(self) -> np.ndarray:
        """Array of upper bounds."""

        return self._upper

    @upper.setter
    def upper(self, upper: np.ndarray) -> None:
        if not isinstance(upper, np.ndarray):
            raise e.TypeError("`upper` should be a numpy array")

        self._upper = upper

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        lower = np.expand_dims(np.expand_dims(space.lb, -1), 0).astype(float)
        self.lower = np.repeat(lower, space.n_agents, axis=0)

        upper = np.expand_dims(np.expand_dims(space.ub, -1), 0).astype(float)
        self.upper = np.repeat(upper, space.n_agents, axis=0)

    def update(self, space: Space, function: Function) -> None:
        """Wraps Cohort Intelligence over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        fitness = [agent.fit for agent in space.agents]

        for i, agent in enumerate(space.agents):
            s = g.weighted_wheel_selection(fitness)

            self.lower[i] = space.agents[s].position - self.lower[i] * self.r / 2
            self.upper[i] = space.agents[s].position - self.upper[i] * self.r / 2

            for _ in range(self.t):
                a = copy.deepcopy(agent)

                for j, (lb, ub) in enumerate(zip(self.lower[i], self.upper[i])):
                    a.position[j] = rnd.generate_uniform_random_number(
                        lb, ub, agent.n_dimensions
                    )
                a.clip_by_bound()

                a.fit = function(a.position)
                if a.fit < agent.fit:
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

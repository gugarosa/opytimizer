"""Differential Evolution.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class DE(Optimizer):
    """A DE class, inherited from Optimizer.

    This is the designed class to define DE-related
    variables and methods.

    References:
        R. Storn. On the usage of differential evolution for function optimization.
        Proceedings of North American Fuzzy Information Processing (1996).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(DE, self).__init__()

        self.CR = 0.9
        self.F = 0.7

        self.build(params)

        logger.info("Class overrided.")

    @property
    def CR(self) -> float:
        """Crossover probability."""

        return self._CR

    @CR.setter
    def CR(self, CR: float) -> None:
        if not isinstance(CR, (float, int)):
            raise e.TypeError("`CR` should be a float or integer")
        if CR < 0 or CR > 1:
            raise e.ValueError("`CR` should be between 0 and 1")

        self._CR = CR

    @property
    def F(self) -> float:
        """Differential weight."""

        return self._F

    @F.setter
    def F(self, F: float) -> None:
        if not isinstance(F, (float, int)):
            raise e.TypeError("`F` should be a float or integer")
        if F < 0 or F > 2:
            raise e.ValueError("`F` should be between 0 and 2")

        self._F = F

    def _mutate_agent(
        self, agent: Agent, alpha: Agent, beta: Agent, gamma: Agent
    ) -> Agent:
        """Mutates a new agent based on pre-picked distinct agents (eq. 4).

        Args:
            agent: Current agent.
            alpha: 1st picked agent.
            beta: 2nd picked agent.
            gamma: 3rd picked agent.

        Returns:
            (Agent): A mutated agent.

        """

        a = copy.deepcopy(agent)

        R = r.generate_integer_random_number(0, agent.n_variables)

        for j in range(a.n_variables):
            r1 = r.generate_uniform_random_number()
            if r1 < self.CR or j == R:
                a.position[j] = alpha.position[j] + self.F * (
                    beta.position[j] - gamma.position[j]
                )

        return a

    def update(self, space: Space, function: Function) -> None:
        """Wraps Differential Evolution over all agents and variables (eq. 1-4).

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        for i, agent in enumerate(space.agents):
            C = d.generate_choice_distribution(
                np.setdiff1d(range(0, len(space.agents)), i), size=3
            )

            a = self._mutate_agent(
                agent, space.agents[C[0]], space.agents[C[1]], space.agents[C[2]]
            )
            a.clip_by_bound()

            a.fit = function(a.position)
            if a.fit < agent.fit:
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

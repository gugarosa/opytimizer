"""Satin Bowerbird Optimizer.
"""

from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class SBO(Optimizer):
    """A SBO class, inherited from Optimizer.

    This is the designed class to define SBO-related
    variables and methods.

    References:
        S. H. S. Moosavi and V. K. Bardsiri.
        Satin bowerbird optimizer: a new optimization algorithm to optimize ANFIS
        for software development effort estimation.
        Engineering Applications of Artificial Intelligence (2017).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the mp_mutation-heuristics.

        """

        super(SBO, self).__init__()

        self.alpha = 0.9
        self.p_mutation = 0.05
        self.z = 0.02

        self.build(params)

        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        """Step size."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        if alpha < 0:
            raise e.ValueError("`alpha` should be >= 0")

        self._alpha = alpha

    @property
    def p_mutation(self) -> float:
        """Probability of mutation."""

        return self._p_mutation

    @p_mutation.setter
    def p_mutation(self, p_mutation: float) -> None:
        if not isinstance(p_mutation, (float, int)):
            raise e.TypeError("`p_mutation` should be a float or integer")
        if p_mutation < 0 or p_mutation > 1:
            raise e.ValueError("`p_mutation` should be between 0 and 1")

        self._p_mutation = p_mutation

    @property
    def z(self) -> float:
        """Percentage of width between lower and upper bounds."""

        return self._z

    @z.setter
    def z(self, z: float) -> None:
        if not isinstance(z, (float, int)):
            raise e.TypeError("`z` should be a float or integer")
        if z < 0 or z > 1:
            raise e.ValueError("`z` should be between 0 and 1")

        self._z = z

    @property
    def sigma(self) -> List[float]:
        """List of widths."""

        return self._sigma

    @sigma.setter
    def sigma(self, sigma: List[float]) -> None:
        if not isinstance(sigma, list):
            raise e.TypeError("`sigma` should be a list")

        self._sigma = sigma

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.sigma = [self.z * (ub - lb) for lb, ub in zip(space.lb, space.ub)]

    def update(self, space: Space, function: Function) -> None:
        """Wraps Satin Bowerbird Optimizer over all agents and variables (eq. 1-7).

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        fitness = [
            1 / (1 + agent.fit) if agent.fit >= 0 else 1 + np.abs(agent.fit)
            for agent in space.agents
        ]
        total_fitness = np.sum(fitness)
        probs = [fit / total_fitness for fit in fitness]

        for agent in space.agents:
            for j in range(agent.n_variables):
                s = d.generate_choice_distribution(len(space.agents), probs, 1)[0]

                lambda_k = self.alpha / (1 + probs[s])

                agent.position[j] += lambda_k * (
                    (space.agents[s].position[j] + space.best_agent.position[j]) / 2
                    - agent.position[j]
                )

                r1 = r.generate_uniform_random_number()
                if r1 < self.p_mutation:
                    agent.position[j] += (
                        self.sigma[j] * r.generate_gaussian_random_number()
                    )
            agent.clip_by_bound()

            agent.fit = function(agent.position)

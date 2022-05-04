"""Flying Squirrel Optimizer.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class FSO(Optimizer):
    """A FSO class, inherited from Optimizer.

    This is the designed class to define FSO-related
    variables and methods.

    References:
        G. Azizyan et al.
        Flying Squirrel Optimizer (FSO): A novel SI-based optimization algorithm for engineering problems.
        Iranian Journal of Optimization (2019).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> FSO.")

        # Overrides its parent class with the receiving params
        super(FSO, self).__init__()

        # Lévy distribution parameter
        self.beta = 0.5

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def beta(self) -> float:
        """Lévy distribution parameter."""

        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if not isinstance(beta, (float, int)):
            raise e.TypeError("`beta` should be a float or integer")
        if beta <= 0 or beta > 2:
            raise e.ValueError("`beta` should be between 0 and 2")

        self._beta = beta

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Flying Squirrel Optimizer over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Calculates the mean position of the population
        mean_position = np.mean([agent.position for agent in space.agents], axis=0)

        # Calculates the Sigma Reduction Factor (eq. 5)
        SRF = (-np.log(1 - (1 / np.sqrt(iteration + 2)))) ** 2

        # Calculates the Beta Expansion Factor
        BEF = self.beta + (2 - self.beta) * ((iteration + 1) / n_iterations)

        # Iterates through all agents
        for agent in space.agents:
            # Makes a deep copy of current agent
            a = copy.deepcopy(agent)

            # Iterates through all variables
            for j in range(agent.n_variables):
                # Calculates the random walk (eq. 2 and 3)
                random_step = r.generate_gaussian_random_number(mean_position[j], SRF)

                # Calculates the Lévy flight (eq. 6 to 18)
                levy_step = d.generate_levy_distribution(BEF)

                # Updates the agent's position
                a.position[j] += (
                    random_step
                    * levy_step
                    * (agent.position[j] - space.best_agent.position[j])
                )

            # Checks agent's limits
            a.clip_by_bound()

            # Re-evaluates the temporary agent
            a.fit = function(a.position)

            # If temporary agent's fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Replace its position and fitness
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

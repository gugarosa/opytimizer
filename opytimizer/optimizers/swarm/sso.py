"""Simplified Swarm Optimization.
"""

import copy
import time
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class SSO(Optimizer):
    """A SSO class, inherited from Optimizer.

    This is the designed class to define SSO-related
    variables and methods.

    References:
        C. Bae et al. A new simplified swarm optimization (SSO) using exchange local search scheme.
        International Journal of Innovative Computing, Information and Control (2012).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> SSO.")

        # Overrides its parent class with the receiving params
        super(SSO, self).__init__()

        # Weighing constant
        self.C_w = 0.1

        # Local constant
        self.C_p = 0.4

        # Global constant
        self.C_g = 0.9

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def C_w(self) -> float:
        """Weighing constant."""

        return self._C_w

    @C_w.setter
    def C_w(self, C_w: float) -> None:
        if not isinstance(C_w, (float, int)):
            raise e.TypeError("`C_w` should be a float or integer")
        if C_w < 0 or C_w > 1:
            raise e.ValueError("`C_w` should be between 0 and 1")

        self._C_w = C_w

    @property
    def C_p(self) -> float:
        """Local constant."""

        return self._C_p

    @C_p.setter
    def C_p(self, C_p: float) -> None:
        if not isinstance(C_p, (float, int)):
            raise e.TypeError("`C_p` should be a float or integer")
        if C_p < self.C_w:
            raise e.ValueError("`C_p` should be equal or greater than `C_w`")

        self._C_p = C_p

    @property
    def C_g(self) -> float:
        """Global constant."""

        return self._C_g

    @C_g.setter
    def C_g(self, C_g: float) -> None:
        if not isinstance(C_g, (float, int)):
            raise e.TypeError("`C_g` should be a float or integer")
        if C_g < self.C_p:
            raise e.ValueError("`C_g` should be equal or greater than `C_p`")

        self._C_g = C_g

    @property
    def local_position(self) -> np.ndarray:
        """Array of local positions."""

        return self._local_position

    @local_position.setter
    def local_position(self, local_position: np.ndarray) -> None:
        if not isinstance(local_position, np.ndarray):
            raise e.TypeError("`local_position` should be a numpy array")

        self._local_position = local_position

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Arrays of local positions
        self.local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def evaluate(self, space: Space, function: Function) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A Space object that will be evaluated.
            function: A Function object that will be used as the objective function.

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Calculates the fitness value of current agent
            fit = function(agent.position)

            # If fitness is better than agent's best fit
            if fit < agent.fit:
                # Updates its current fitness to the newer one
                agent.fit = fit

                # Also updates the local best position to current's agent position
                self.local_position[i] = copy.deepcopy(agent.position)

            # If agent's fitness is better than global fitness
            if agent.fit < space.best_agent.fit:
                # Makes a deep copy of agent's local best position and fitness to the best agent
                space.best_agent.position = copy.deepcopy(self.local_position[i])
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: Space) -> None:
        """Wraps Simplified Swarm Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Iterates through every decision variable
            for j in range(agent.n_variables):
                # Generates a uniform random number
                r1 = r.generate_uniform_random_number()

                # If random number is smaller than `C_w`
                if r1 < self.C_w:
                    # Ignores the position update
                    pass

                # If random number is between `C_w` and `C_p`
                elif r1 < self.C_p:
                    # Updates agent's position with its local position
                    agent.position[j] = self.local_position[i][j]

                # If random number is between `C_p` and `C_g`
                elif r1 < self.C_g:
                    # Updates agent's position with best position
                    agent.position[j] = space.best_agent.position[j]

                # If random number is greater than `C_g`
                else:
                    # Updates agent's position with random number
                    agent.position[j] = r.generate_uniform_random_number(
                        size=agent.n_dimensions
                    )

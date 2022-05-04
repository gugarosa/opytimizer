"""Boolean Particle Swarm Optimization.
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


class BPSO(Optimizer):
    """A BPSO class, inherited from Optimizer.

    This is the designed class to define boolean PSO-related
    variables and methods.

    References:
        F. Afshinmanesh, A. Marandi and A. Rahimi-Kian.
        A Novel Binary Particle Swarm Optimization Method Using Artificial Immune System.
        IEEE International Conference on Smart Technologies (2005).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> BPSO.")

        # Overrides its parent class with the receiving params
        super(BPSO, self).__init__()

        # Cognitive constant
        self.c1 = np.array([1])

        # Social constant
        self.c2 = np.array([1])

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def c1(self) -> np.ndarray:
        """Cognitive constant."""

        return self._c1

    @c1.setter
    def c1(self, c1: np.ndarray) -> None:
        if not isinstance(c1, np.ndarray):
            raise e.TypeError("`c1` should be a numpy array")

        self._c1 = c1

    @property
    def c2(self) -> np.ndarray:
        """Social constant."""

        return self._c2

    @c2.setter
    def c2(self, c2: np.ndarray) -> None:
        if not isinstance(c2, np.ndarray):
            raise e.TypeError("`c2` should be a numpy array")

        self._c2 = c2

    @property
    def local_position(self) -> np.ndarray:
        """Array of local positions."""

        return self._local_position

    @local_position.setter
    def local_position(self, local_position: np.ndarray) -> None:
        if not isinstance(local_position, np.ndarray):
            raise e.TypeError("`local_position` should be a numpy array")

        self._local_position = local_position

    @property
    def velocity(self) -> np.ndarray:
        """Array of velocities."""

        return self._velocity

    @velocity.setter
    def velocity(self, velocity: np.ndarray) -> None:
        if not isinstance(velocity, np.ndarray):
            raise e.TypeError("`velocity` should be a numpy array")

        self._velocity = velocity

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Arrays of local positions and velocities
        self.local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions), dtype=bool
        )
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions), dtype=bool
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
        """Wraps Boolean Particle Swarm Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Defines random binary numbers
            r1 = r.generate_binary_random_number(agent.position.shape)
            r2 = r.generate_binary_random_number(agent.position.shape)

            # Calculates the local and global partials
            local_partial = np.logical_and(
                self.c1,
                np.logical_xor(
                    r1, np.logical_xor(self.local_position[i], agent.position)
                ),
            )
            global_partial = np.logical_and(
                self.c2,
                np.logical_xor(
                    r2, np.logical_xor(space.best_agent.position, agent.position)
                ),
            )

            # Updates current agent velocities (eq. 1)
            self.velocity[i] = np.logical_or(local_partial, global_partial)

            # Updates current agent positions (eq. 2)
            agent.position = np.logical_xor(agent.position, self.velocity[i])

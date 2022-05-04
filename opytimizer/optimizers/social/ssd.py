"""Social Ski Driver.
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


class SSD(Optimizer):
    """An SSD class, inherited from Optimizer.

    This is the designed class to define SSD-related
    variables and methods.

    References:
        A. Tharwat and T. Gabel.
        Parameters optimization of support vector machines for imbalanced data using social ski driver algorithm.
        Neural Computing and Applications (2019).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> SSD.")

        # Overrides its parent class with the receiving params
        super(SSD, self).__init__()

        # Exploration parameter
        self.c = 2.0

        # Decay rate
        self.decay = 0.99

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def c(self) -> float:
        """Exploration parameter."""

        return self._c

    @c.setter
    def c(self, c: float) -> None:
        if not isinstance(c, (float, int)):
            raise e.TypeError("`c` should be a float or integer")
        if c < 0:
            raise e.ValueError("`c` should be >= 0")

        self._c = c

    @property
    def decay(self) -> float:
        """Decay rate."""

        return self._decay

    @decay.setter
    def decay(self, decay: float) -> None:
        if not isinstance(decay, (float, int)):
            raise e.TypeError("`decay` should be a float or integer")
        if decay < 0 or decay > 1:
            raise e.ValueError("`decay` should be between 0 and 1")
        self._decay = decay

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
            (space.n_agents, space.n_variables, space.n_dimensions)
        )
        self.velocity = r.generate_uniform_random_number(
            size=(space.n_agents, space.n_variables, space.n_dimensions)
        )

    def _mean_global_solution(
        self, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray
    ) -> np.ndarray:
        """Calculates the mean global solution (eq. 9).

        Args:
            alpha: 1st agent's current position.
            beta: 2nd agent's current position.
            gamma: 3rd agent's current position.

        Returns:
            (np.ndarray): Mean global solution.

        """

        # Calculates the mean global solution
        mean = (alpha + beta + gamma) / 3

        return mean

    def _update_position(self, position: np.ndarray, index: int) -> np.ndarray:
        """Updates a particle position (eq. 10).

        Args:
            position: Agent's current position.
            index: Index of current agent.

        Returns:
            (np.ndarray): A new position.

        """

        # Calculates new position
        new_position = position + self.velocity[index]

        return new_position

    def _update_velocity(
        self, position: np.ndarray, mean: np.ndarray, index: int
    ) -> np.ndarray:
        """Updates a particle velocity (eq. 11).

        Args:
            position: Agent's current position.
            mean: Mean global best position.
            index: Index of current agent.

        Returns:
            (np.ndarray): A new velocity.

        """

        # Generates random numbers
        r1 = r.generate_uniform_random_number()
        r2 = r.generate_uniform_random_number()

        # If random number is smaller than or equal to 0.5
        if r2 <= 0.5:
            # Updates its velocity based on sine wave
            new_velocity = self.c * np.sin(r1) * (
                self.local_position[index] - position
            ) + np.sin(r1) * (mean - position)

        # If random number is bigger than 0.5
        else:
            # Updates its velocity based on cosine wave
            new_velocity = self.c * np.cos(r1) * (
                self.local_position[index] - position
            ) + np.cos(r1) * (mean - position)

        return new_velocity

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

    def update(self, space: Space, function: Function) -> None:
        """Wraps Social Ski Driver over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Calculates the new fitness
            fit = function(agent.position)

            # If new fitness is better than agent's fitness
            if fit < agent.fit:
                # Updates its current fitness to the newer one
                agent.fit = fit

                # Also updates the local best position to current agent's position
                self.local_position[i] = copy.deepcopy(agent.position)

            # Sorts agents
            space.agents.sort(key=lambda x: x.fit)

            # Calculates the mean global solution
            mean = self._mean_global_solution(
                space.agents[0].position,
                space.agents[1].position,
                space.agents[2].position,
            )

            # Updates current agent positions
            agent.position = self._update_position(agent.position, i)

            # Checks agent limits
            agent.clip_by_bound()

            # Updates current agent velocities
            self.velocity[i] = self._update_velocity(agent.position, mean, i)

        # Reduces exploration parameter
        self.c *= self.decay

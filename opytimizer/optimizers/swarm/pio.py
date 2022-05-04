"""Pigeon-Inspired Optimization.
"""

from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class PIO(Optimizer):
    """A PIO class, inherited from Optimizer.

    This is the designed class to define PIO-related
    variables and methods.

    References:
        H. Duan and P. Qiao.
        Pigeon-inspired optimization:a new swarm intelligence optimizerfor air robot path planning.
        International Journal of IntelligentComputing and Cybernetics (2014).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> PIO.")

        # Overrides its parent class with the receiving params
        super(PIO, self).__init__()

        # Number of mapping iterations
        self.n_c1 = 150

        # Number of landmark iterations
        self.n_c2 = 200

        # Map and compass factor
        self.R = 0.2

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def n_c1(self) -> int:
        """Number of mapping iterations."""

        return self._n_c1

    @n_c1.setter
    def n_c1(self, n_c1: int) -> None:
        if not isinstance(n_c1, int):
            raise e.TypeError("`n_c1` should be an integer")
        if n_c1 <= 0:
            raise e.ValueError("`n_c1` should be > 0")

        self._n_c1 = n_c1

    @property
    def n_c2(self) -> int:
        """Number of landmark iterations."""

        return self._n_c2

    @n_c2.setter
    def n_c2(self, n_c2: int) -> None:
        if not isinstance(n_c2, int):
            raise e.TypeError("`n_c2` should be an integer")
        if n_c2 < self.n_c1:
            raise e.ValueError("`n_c1` should be > `n_c2")

        self._n_c2 = n_c2

    @property
    def R(self) -> float:
        """Map and compass factor."""

        return self._R

    @R.setter
    def R(self, R: float) -> None:
        if not isinstance(R, (float, int)):
            raise e.TypeError("`R` should be a float or integer")
        if R < 0:
            raise e.ValueError("`R` should be >= 0")

        self._R = R

    @property
    def n_p(self) -> int:
        """Number of pigeons."""

        return self._n_p

    @n_p.setter
    def n_p(self, n_p: int) -> None:
        if not isinstance(n_p, int):
            raise e.TypeError("`n_p` should be an integer")
        if n_p <= 0:
            raise e.ValueError("`n_p` should be > 0")

        self._n_p = n_p

    @property
    def velocity(self) -> np.ndarray:
        """Array of pulse rates."""

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

        # Number of pigeons
        self.n_p = space.n_agents

        # Array of velocities
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def _calculate_center(self, agents: List[Agent]) -> np.ndarray:
        """Calculates the center position (eq. 8).

        Args:
            agents: List of agents.

        Returns:
            (np.ndarray): The center position.

        """

        # Creates an array to hold the cummulative position
        total_pos = np.zeros((agents[0].n_variables, agents[0].n_dimensions))

        # Initializes total fitness as zero
        total_fit = 0.0

        # Iterates through all agents
        for agent in agents:
            # Accumulates the position
            total_pos += agent.position * agent.fit

            # Accumulates the fitness
            total_fit += agent.fit

        # Calculates the center position
        center = total_pos / (self.n_p * total_fit + c.EPSILON)

        return center

    def _update_center_position(self, position: np.ndarray, center: np.ndarray) -> None:
        """Updates a pigeon position based on the center (eq. 9).

        Args:
            position: Agent's current position.
            center: Center position.

        Returns:
            (np.ndarray): A new center-based position.

        """

        # Generates random number
        r1 = r.generate_uniform_random_number()

        # Calculates new position based on center
        new_position = position + r1 * (center - position)

        return new_position

    def update(self, space: Space, iteration: int) -> None:
        """Wraps Pigeon-Inspired Optimization over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.

        """

        # Checks if current iteration is smaller than mapping operator
        if iteration < self.n_c1:
            # Iterates through all agents
            for i, agent in enumerate(space.agents):
                # Updates current agent velocity (eq. 5)
                r1 = r.generate_uniform_random_number()
                self.velocity[i] = self.velocity[i] * np.exp(
                    -self.R * (iteration + 1)
                ) + r1 * (space.best_agent.position - agent.position)

                # Updates current agent position (eq. 6)
                agent.position += self.velocity[i]

        # Checks if current iteration is smaller than landmark operator
        elif iteration < self.n_c2:
            # Calculates the number of possible pigeons (eq. 7)
            self.n_p = int(self.n_p / 2) + 1

            # Sorts agents according to their fitness
            space.agents.sort(key=lambda x: x.fit)

            # Calculates the center position
            center = self._calculate_center(space.agents[: self.n_p])

            # Iterates through all agents
            for agent in space.agents:
                # Updates current agent position
                agent.position = self._update_center_position(agent.position, center)

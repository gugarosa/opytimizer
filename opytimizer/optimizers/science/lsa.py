"""Lightning Search Algorithm.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class LSA(Optimizer):
    """An LSA class, inherited from Optimizer.

    This is the designed class to define LSA-related
    variables and methods.

    References:
        H. Shareef, A. Ibrahim and A. Mutlag. Lightning search algorithm.
        Applied Soft Computing (2015).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> LSA.")

        # Overrides its parent class with the receiving params
        super(LSA, self).__init__()

        # Maximum channel time
        self.max_time = 10

        # Initial energy
        self.E = 2.05

        # Forking probability
        self.p_fork = 0.01

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def max_time(self) -> int:
        """Maximum channel time."""

        return self._max_time

    @max_time.setter
    def max_time(self, max_time: int) -> None:
        if not isinstance(max_time, int):
            raise e.TypeError("`max_time` should be an integer")
        if max_time <= 0:
            raise e.ValueError("`max_time` should be > 0")

        self._max_time = max_time

    @property
    def E(self) -> float:
        """Initial energy."""

        return self._E

    @E.setter
    def E(self, E: float) -> None:
        if not isinstance(E, (float, int)):
            raise e.TypeError("`E` should be a float or integer")
        if E < 0:
            raise e.ValueError("`E` should be >= 0")

        self._E = E

    @property
    def p_fork(self) -> float:
        """Probability of forking."""

        return self._p_fork

    @p_fork.setter
    def p_fork(self, p_fork: float) -> None:
        if not isinstance(p_fork, (float, int)):
            raise e.TypeError("`p_fork` should be a float or integer")
        if p_fork < 0 or p_fork > 1:
            raise e.ValueError("`p_fork` should be between 0 and 1")

        self._p_fork = p_fork

    @property
    def time(self) -> int:
        """Channel time."""

        return self._time

    @time.setter
    def time(self, time: int) -> None:
        if not isinstance(time, int):
            raise e.TypeError("`time` should be an integer")
        if time < 0:
            raise e.ValueError("`time` should be >= 0")

        self._time = time

    @property
    def direction(self) -> np.ndarray:
        """Array of directions."""

        return self._direction

    @direction.setter
    def direction(self, direction: np.ndarray) -> np.ndarray:
        if not isinstance(direction, np.ndarray):
            raise e.TypeError("`direction` should be a numpy array")

        self._direction = direction

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Channel time
        self.time = 0

        # Array of directions
        self.direction = np.sign(
            r.generate_uniform_random_number(
                -1, 1, (space.n_variables, space.n_dimensions)
            )
        )

    def _update_direction(self, agent: Agent, function: Function) -> None:
        """Updates the direction array by shaking agent's direction.

        Args:
            agent: An agent instance.
            function: A Function object that will be used as the objective function.

        """

        # Iterates through all decision variables
        for j in range(agent.n_variables):
            # Makes a copy of agent
            direction = copy.deepcopy(agent)

            # Shakes the direction
            direction.position[j] += (
                self.direction[j] * 0.005 * (agent.ub[j] - agent.lb[j])
            )

            # Clips its bounds
            direction.clip_by_bound()

            # Evaluates the direction
            direction.fit = function(direction.position)

            # If new direction's fitness is worst than agent's fitness
            if direction.fit > agent.fit:
                # Inverts the direction
                self.direction[j] *= -1

    def _update_position(
        self, agent: Agent, best_agent: Agent, function: Function, energy: float
    ) -> None:
        """Updates agent's position.

        Args:
            agent: An agent instance.
            best_agent: A best agent instance.
            function: A Function object that will be used as the objective function.
            energy: Current energy value.

        """

        # Makes a copy of agent
        a = copy.deepcopy(agent)

        # Calculates the distance between agent and best agent
        distance = agent.position - best_agent.position

        # Iterates through all decision variables
        for j in range(agent.n_variables):
            # Iterates through all dimensions
            for k in range(agent.n_dimensions):
                # If distance equals to zero
                if distance[j][k] == 0:
                    # Updates the position by sampling a gaussian number
                    r1 = r.generate_gaussian_random_number(0, energy)
                    a.position[j][k] += self.direction[j][k] * r1

                # If distance is different from zero
                else:
                    # If distance is smaller than zero
                    if distance[j][k] < 0:
                        # Updates the position by adding an exponential number
                        a.position[j][k] += r.generate_exponential_random_number(
                            np.fabs(distance[j][k])
                        )

                    # If distance is bigger than zero
                    else:
                        # Updates the position by subtracting an exponential number
                        a.position[j][k] -= r.generate_exponential_random_number(
                            distance[j][k]
                        )

        # Clips the temporary agent's limits
        a.clip_by_bound()

        # Evaluates its new position
        a.fit = function(a.position)

        # If temporary agent's fitness is better than current agent's fitness
        if a.fit < agent.fit:
            # Replaces position and fitness
            agent.position = copy.deepcopy(a.position)
            agent.fit = copy.deepcopy(a.fit)

            # Generates a random number
            r1 = r.generate_uniform_random_number()

            # If random number is smaller than probability of forking
            if r1 < self.p_fork:
                # Makes a new copy of current agent
                a = copy.deepcopy(agent)

                # Generates a random position
                a.fill_with_uniform()

                # Re-evaluates its position
                a.fit = function(a.position)

                # If new fitness is better than agent's fitness
                if a.fit < agent.fit:
                    # Replaces position and fitness
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

    def update(
        self, space: Space, function: Function, iteration: int, n_iterations: int
    ) -> None:
        """Wraps Lightning Search Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Increases the channel's time
        self.time += 1

        # If channel has reached maximum allowed time
        if self.time >= self.max_time:
            # Sorts agents
            space.agents.sort(key=lambda x: x.fit)

            # Replaces the worst channel with the best one
            space.agents[-1] = copy.deepcopy(space.agents[0])

            # Resets the channel's time
            self.time = 0

        # Re-sorts the agents
        space.agents.sort(key=lambda x: x.fit)

        # Updates the direction
        self._update_direction(space.agents[0], function)

        # Calculates the current energy
        energy = self.E - 2 * np.exp(-5 * (n_iterations - iteration) / n_iterations)

        # Iterates through all agents
        for agent in space.agents:
            # Updates agent's position
            self._update_position(agent, space.agents[0], function, energy)

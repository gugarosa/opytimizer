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

        super(LSA, self).__init__()

        self.max_time = 10
        self.E = 2.05
        self.p_fork = 0.01

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

        self.time = 0

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

        for j in range(agent.n_variables):
            direction = copy.deepcopy(agent)
            direction.position[j] += (
                self.direction[j] * 0.005 * (agent.ub[j] - agent.lb[j])
            )
            direction.clip_by_bound()

            direction.fit = function(direction.position)
            if direction.fit > agent.fit:
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

        a = copy.deepcopy(agent)

        distance = agent.position - best_agent.position

        for j in range(agent.n_variables):
            for k in range(agent.n_dimensions):
                if distance[j][k] == 0:
                    r1 = r.generate_gaussian_random_number(0, energy)
                    a.position[j][k] += self.direction[j][k] * r1
                else:
                    if distance[j][k] < 0:
                        a.position[j][k] += r.generate_exponential_random_number(
                            np.fabs(distance[j][k])
                        )
                    else:
                        a.position[j][k] -= r.generate_exponential_random_number(
                            distance[j][k]
                        )
        a.clip_by_bound()

        a.fit = function(a.position)
        if a.fit < agent.fit:
            agent.position = copy.deepcopy(a.position)
            agent.fit = copy.deepcopy(a.fit)

            r1 = r.generate_uniform_random_number()
            if r1 < self.p_fork:
                a = copy.deepcopy(agent)
                a.fill_with_uniform()

                a.fit = function(a.position)
                if a.fit < agent.fit:
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

        self.time += 1
        if self.time >= self.max_time:
            space.agents.sort(key=lambda x: x.fit)
            space.agents[-1] = copy.deepcopy(space.agents[0])

            self.time = 0

        space.agents.sort(key=lambda x: x.fit)

        self._update_direction(space.agents[0], function)

        energy = self.E - 2 * np.exp(-5 * (n_iterations - iteration) / n_iterations)

        for agent in space.agents:
            self._update_position(agent, space.agents[0], function, energy)

"""Runner-Root Algorithm.
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class RRA(Optimizer):
    """An RRA class, inherited from Optimizer.

    This is the designed class to define RRA-related
    variables and methods.

    References:
        F. Merrikh-Bayat.
        The runner-root algorithm: A metaheuristic for solving unimodal and
        multimodal optimization problems inspired by runners and roots of plants in nature.
        Applied Soft Computing (2015).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> RRA.")

        # Overrides its parent class with the receiving params
        super(RRA, self).__init__()

        # Length of runners
        self.d_runner = 2

        # Length of roots
        self.d_root = 0.01

        # Cost function tolerance
        self.tol = 0.01

        # Maximum number of stalls
        self.max_stall = 1000

        # Current number of stalls
        self.n_stall = 0

        # Previous best fitness value
        self.last_best_fit = c.FLOAT_MAX

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def d_runner(self) -> int:
        """Length of runners."""

        return self._d_runner

    @d_runner.setter
    def d_runner(self, d_runner: int) -> None:
        if not isinstance(d_runner, int):
            raise e.TypeError("`d_runner` should be an integer")
        if d_runner <= 0:
            raise e.ValueError("`d_runner` should be > 0")

        self._d_runner = d_runner

    @property
    def d_root(self) -> float:
        """Length of roots."""

        return self._d_root

    @d_root.setter
    def d_root(self, d_root: float) -> None:
        if not isinstance(d_root, (float, int)):
            raise e.TypeError("`d_root` should be a float or integer")
        if d_root < 0:
            raise e.ValueError("`d_root` should be >= 0")

        self._d_root = d_root

    @property
    def tol(self) -> float:
        """Cost function tolerance."""

        return self._tol

    @tol.setter
    def tol(self, tol: float) -> None:
        if not isinstance(tol, (float, int)):
            raise e.TypeError("`tol` should be a float or integer")
        if tol < 0:
            raise e.ValueError("`tol` should be >= 0")

        self._tol = tol

    @property
    def max_stall(self) -> int:
        """Maximum number of stalls."""

        return self._max_stall

    @max_stall.setter
    def max_stall(self, max_stall: int) -> None:
        if not isinstance(max_stall, int):
            raise e.TypeError("`max_stall` should be an integer")
        if max_stall <= 0:
            raise e.ValueError("`max_stall` should be > 0")

        self._max_stall = max_stall

    @property
    def n_stall(self) -> int:
        """Current number of stalls."""

        return self._n_stall

    @n_stall.setter
    def n_stall(self, n_stall: int) -> None:
        if not isinstance(n_stall, int):
            raise e.TypeError("`n_stall` should be an integer")
        if n_stall < 0:
            raise e.ValueError("`n_stall` should be > 0")
        if n_stall > self.max_stall:
            raise e.ValueError("`n_stall` should be smaller than `max_stall")

        self._n_stall = n_stall

    @property
    def last_best_fit(self) -> float:
        """Previous best fitness value."""

        return self._last_best_fit

    @last_best_fit.setter
    def last_best_fit(self, last_best_fit: float) -> None:
        if not isinstance(last_best_fit, (float, int)):
            raise e.TypeError("`last_best_fit` should be a float or integer")

        self._last_best_fit = last_best_fit

    def _stalling_search(
        self,
        daughters: List[Agent],
        function: Function,
        is_large: Optional[bool] = True,
    ) -> None:
        """Performs the stalling random larrge or small search (eq. 4 and 5).

        Args:
            daughters: Daughters.
            function: A Function object that will be used as the objective function.
            is_large: Whether to perform the large or small search.

        """

        # Iterates through number of daughters minus 1
        for _ in range(len(daughters) - 1):
            # Makes a deep copy of best daughter
            temp_daughter = copy.deepcopy(daughters[0])

            # Generates gaussian random number and variable's index
            j = r.generate_integer_random_number(high=temp_daughter.n_variables)

            # Checks whether to perform the large search
            if is_large:
                # Disturbs a selected temporary daughter's position (eq. 4)
                r1 = r.generate_gaussian_random_number()
                temp_daughter.position[j] += self.d_runner * r1
            else:
                # Disturbs a selected temporary daughter's position (eq. 5)
                r1 = r.generate_uniform_random_number(-0.5, 0.5)
                temp_daughter.position[j] += self.d_root * r1

            # Clips the temporary daughter's position
            temp_daughter.clip_by_bound()

            # Re-evaluates its fitness
            temp_daughter.fit = function(temp_daughter.position)

            # If temporary daughter's position is better than best's
            if temp_daughter.fit < daughters[0].fit:
                # Replaces the best daughter
                daughters[0].position = copy.deepcopy(temp_daughter.position)
                daughters[0].fit = copy.deepcopy(temp_daughter.fit)

    def _roulette_selection(
        self, fitness: List[float], a: Optional[float] = 0.1
    ) -> int:
        """Performs a roulette selection on the population (eq. 8).

        Args:
            fitness: A fitness list of every agent.
            a: Selection regularizer.

        Returns:
            (int): The selected index of the population.

        """

        # Defines the minimum fitness of current generation
        min_fitness = np.min(fitness)

        # Re-arrange the list of fitness by inverting it (eq. 7)
        inv_fitness = [1 / (a + fit - min_fitness) for fit in fitness]

        # Calculates the total inverted fitness
        total_fitness = np.sum(inv_fitness)

        # Calculates the probability of each inverted fitness (eq. 8)
        probs = [fit / total_fitness for fit in inv_fitness]

        # Performs the selection process
        selected = d.generate_choice_distribution(len(probs), probs, 1)

        return selected[0]

    def update(self, space: Space, function: Function) -> None:
        """Wraps Runner-Root Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Sorts the agents
        space.agents.sort(key=lambda x: x.fit)

        # Defines the previous best fitness
        self.last_best_fit = space.agents[0].fit

        # Makes a deep copy of current agents (mothers)
        daughters = copy.deepcopy(space.agents)

        # Iterates through all daughters except the first
        for daughter in daughters[1:]:
            # Generates a random number
            r1 = r.generate_uniform_random_number(-0.5, 0.5)

            # Updates the daughter's position and clips its bounds (eq. 2)
            daughter.position += self.d_runner * r1
            daughter.clip_by_bound()

            # Re-calculates its fitness
            daughter.fit = function(daughter.position)

        # Sorts the daughters
        daughters.sort(key=lambda x: x.fit)

        # Checks the new positions' effectiviness (eq. 3)
        effectiveness = np.fabs(
            (self.last_best_fit - daughters[0].fit) / (self.last_best_fit + c.EPSILON)
        )

        # If effectiveness is smaller than tolerance
        if effectiveness < self.tol:
            # Performs the stalling large search (eq. 4)
            self._stalling_search(daughters, function, is_large=True)

            # Performs the stalling small search (eq. 5)
            self._stalling_search(daughters, function, is_large=False)

        # Performs the elite selection (eq. 6)
        space.agents[0] = copy.deepcopy(daughters[0])

        # Gathers the fitness of each daughter
        daughters_fit = [daughter.fit for daughter in daughters]

        # Iterates through all mothers except the first
        for agent in space.agents[1:]:
            # Gathers the selected daughter by roulette selection
            idx = self._roulette_selection(daughters_fit)

            # Replaces the mother with the daughter
            agent = copy.deepcopy(daughters[idx])

        # Checks again the positions' effectiviness (eq. 3)
        effectiveness = np.fabs(
            (self.last_best_fit - daughters[0].fit) / (self.last_best_fit + c.EPSILON)
        )

        # If effectiveness is smaller than tolerance
        if effectiveness < self.tol:
            # Increases the number of stalls
            self.n_stall += 1
        else:
            # Resets the number of stalls
            self.n_stall = 0

        # If the number of stalls has reached its maximum
        if self.n_stall == self.max_stall:
            # Iterates through all agenrs
            for agent in space.agents:
                # Fills the agent with random positions
                agent.fill_with_uniform()

            # Resets the counter
            self.n_stall = 0

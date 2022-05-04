"""Parasitism-Predation Algorithm.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class PPA(Optimizer):
    """A PPA class, inherited from Optimizer.

    This is the designed class to define PPA-related
    variables and methods.

    References:
        A. Mohamed et al. Parasitism – Predation algorithm (PPA): A novel approach for feature selection.
        Ain Shams Engineering Journal (2020).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> PPA.")

        # Overrides its parent class with the receiving params
        super(PPA, self).__init__()

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

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

        # Array of velocities
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions)
        )

    def _calculate_population(
        self, n_agents: int, iteration: int, n_iterations: int
    ) -> Tuple[int, int, int]:
        """Calculates the number of crows, cats and cuckoos.

        Args:
            n_agents: Number of agents.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        Returns:
            (Tuple[int, int, int]): The number of crows, cats and cuckoos.

        """

        # Calculates the number of crows
        n_crows = np.round(
            n_agents * (2 / 3 - iteration * ((2 / 3 - 1 / 2) / n_iterations))
        )

        # Calculates the number of cats
        n_cats = np.round(
            n_agents * (0.01 + iteration * ((1 / 3 - 0.01) / n_iterations))
        )

        # Calculates the number of cuckoos
        n_cuckoos = n_agents - n_crows - n_cats

        return int(n_crows), int(n_cats), int(n_cuckoos)

    def _nesting_phase(self, space: Space, n_crows: int):
        """Performs the nesting phase using the current number of crows.

        Args:
            space: Space containing agents and update-related information.
            n_crows: Number of crows.

        """

        # Gathers the crows
        crows = space.agents[:n_crows]

        # Iterates through all crows
        for i, crow in enumerate(crows):
            # Generates a random index
            idx = r.generate_integer_random_number(high=space.n_agents, exclude_value=i)

            # Calculates the step from Lévy distribution (eq. 7)
            step = d.generate_levy_distribution(size=crow.n_variables)
            step = np.expand_dims(step, axis=1)

            # Updates the crow's position and clips its bounds (eq. 6 and 8)
            crow.position = 0.01 * step * (space.agents[idx].position - crow.position)
            crow.clip_by_bound()

    def _parasitism_phase(
        self,
        space: Space,
        n_crows: int,
        n_cuckoos: int,
        iteration: int,
        n_iterations: int,
    ):
        """Performs the parasitism phase using the current number of cuckoos.

        Args:
            space: Space containing agents and update-related information.
            n_crows: Number of crows.
            n_cuckoos: Number of cuckoos.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Gathers the cuckoos
        cuckoos = space.agents[n_crows : n_crows + n_cuckoos]

        # Calculates a list of cuckoos' fitness
        fitness = [cuckoo.fit for cuckoo in cuckoos]

        # Calculates the probability of selection
        p = iteration / n_iterations

        # Iterates through all cuckoos
        for cuckoo in cuckoos:
            # Selects a cuckoo through tournament selection
            s = g.tournament_selection(fitness, 1)[0]

            # Selects two random agents from the space
            i = r.generate_integer_random_number(high=space.n_agents)
            j = r.generate_integer_random_number(high=space.n_agents, exclude_value=i)

            # Creates a bernoulli distribution to preserve or not variables (eq. 12)
            k = d.generate_bernoulli_distribution(1 - p, cuckoo.n_variables)
            k = np.expand_dims(k, -1)

            # Calculates the gaussian-based step distribution (eq. 11)
            rand = r.generate_uniform_random_number()
            S_g = (space.agents[i].position - space.agents[j].position) * rand

            # Updates the cuckoo's position and clips its limits (eq. 10)
            cuckoo.position = space.agents[s].position + S_g * k
            cuckoo.clip_by_bound()

    def _predation_phase(
        self,
        space: Space,
        n_crows: int,
        n_cuckoos: int,
        n_cats: int,
        iteration: int,
        n_iterations: int,
    ) -> None:
        """Performs the predation phase using the current number of cats.

        Args:
            space: Space containing agents and update-related information.
            n_crows: Number of crows.
            n_cuckoos: Number of cuckoos.
            n_cats: Number of cats.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Gathers the cats
        cats = space.agents[n_crows + n_cuckoos :]

        # Calculates the constant
        constant = 2 - iteration / n_iterations

        # Iterates through all cats
        for i, cat in enumerate(cats):
            # Gets the corresponding cat's index
            idx = space.n_agents - n_cats + i

            # Updates the cat's velocity (eq. 13)
            r1 = r.generate_uniform_random_number()
            self.velocity[idx] += (
                r1 * constant * (space.best_agent.position - cat.position)
            )

            # Updates the cat's position and clips its limits (eq. 14)
            cat.position += self.velocity[idx]
            cat.clip_by_bound()

    def update(self, space: Space, iteration: int, n_iterations: int) -> None:
        """Wraps Parasitism-Predation Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            iteration: Current iteration.
            n_iterations: Maximum number of iterations.

        """

        # Calculates the number of crows, cats and cuckoos
        n_crows, n_cats, n_cuckoos = self._calculate_population(
            space.n_agents, iteration, n_iterations
        )

        # Performs the nesting phase
        self._nesting_phase(space, n_crows)

        # Performs the parasitism phase
        self._parasitism_phase(space, n_crows, n_cuckoos, iteration, n_iterations)

        # Performs the predation phase
        self._predation_phase(
            space, n_crows, n_cuckoos, n_cats, iteration, n_iterations
        )

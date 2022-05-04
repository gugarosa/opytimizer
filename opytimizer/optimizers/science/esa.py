"""Electro-Search Algorithm.
"""

import copy
from typing import Any, Dict, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class ESA(Optimizer):
    """An ESA class, inherited from Optimizer.

    This is the designed class to define ES-related
    variables and methods.

    References:
        A. Tabari and A. Ahmad. A new optimization method: Electro-Search algorithm.
        Computers & Chemical Engineering (2017).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> ESA.")

        # Overrides its parent class with the receiving params
        super(ESA, self).__init__()

        # Number of electrons per atom
        self.n_electrons = 5

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def n_electrons(self) -> int:
        """Number of electrons per atom."""

        return self._n_electrons

    @n_electrons.setter
    def n_electrons(self, n_electrons: int) -> None:
        if not isinstance(n_electrons, int):
            raise e.TypeError("`n_electrons` should be an integer")
        if n_electrons <= 0:
            raise e.ValueError("`n_electrons` should be > 0")

        self._n_electrons = n_electrons

    @property
    def D(self) -> np.ndarray:
        """Orbital radius."""

        return self._D

    @D.setter
    def D(self, D: np.ndarray) -> None:
        if not isinstance(D, np.ndarray):
            raise e.TypeError("`D` should be a numpy array")

        self._D = D

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        # Orbital radius
        self.D = r.generate_uniform_random_number(
            size=(space.n_agents, space.n_variables, space.n_dimensions)
        )

    def update(self, space: Space, function: Function) -> None:
        """Wraps EElectro-Search Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Makes a deep copy of current agent
            a = copy.deepcopy(agent)

            # Creates a list of electrons
            electrons = [copy.deepcopy(agent) for _ in range(self.n_electrons)]

            # Iterates through all electrons
            for electron in electrons:
                # Generates a random number and the energy level
                r1 = r.generate_uniform_random_number()
                n = r.generate_integer_random_number(2, 6)

                # Updates the electron's position (eq. 3)
                electron.position += (2 * r1 - 1) * (1 - 1 / n**2) / self.D[i]

                # Clips its bounds
                electron.clip_by_bound()

                # Re-evaluates the new position
                electron.fit = function(electron.position)

            # Sorts the electrons
            electrons.sort(key=lambda x: x.fit)

            # Generates both Rydberg constant and acceleration coefficient
            # Original implementation is missing up an informative description
            Re = r.generate_uniform_random_number()
            Ac = r.generate_uniform_random_number()

            # Updates the Orbital radius (eq. 4)
            self.D[i] = (electrons[0].position - space.best_agent.position) + Re * (
                1 / space.best_agent.position**2 - 1 / a.position**2
            )

            # Updates the temporary agent's position (eq. 5)
            a.position += Ac * self.D[i]

            # Checks agent's limits
            a.clip_by_bound()

            # Calculates the fitness for the temporary position
            a.fit = function(a.position)

            # If new fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Copies its position and fitness to the agent
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)

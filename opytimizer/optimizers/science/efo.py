"""Electromagnetic Field Optimization.
"""

import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class EFO(Optimizer):
    """An EFO class, inherited from Optimizer.

    This is the designed class to define EFO-related
    variables and methods.

    References:
        H. Abedinpourshotorban et al.
        Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm.
        Swarm and Evolutionary Computation (2016).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        # Overrides its parent class with the receiving params
        super(EFO, self).__init__()

        # Positive field proportion
        self.positive_field = 0.1

        # Negative field proportion
        self.negative_field = 0.5

        # Probability of selecting eletromagnets
        self.ps_ratio = 0.1

        # Probability of selecting a random eletromagnet
        self.r_ratio = 0.4

        # Golden ratio
        self.phi = (1 + np.sqrt(5)) / 2

        # Eletromagnetic index
        self.RI = 0

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def positive_field(self) -> float:
        """Positive field proportion."""

        return self._positive_field

    @positive_field.setter
    def positive_field(self, positive_field: float) -> None:
        if not isinstance(positive_field, (float, int)):
            raise e.TypeError("`positive_field` should be a float or integer")
        if positive_field < 0 or positive_field > 1:
            raise e.ValueError("`positive_field` should be between 0 and 1")

        self._positive_field = positive_field

    @property
    def negative_field(self) -> float:
        """Negative field proportion."""

        return self._negative_field

    @negative_field.setter
    def negative_field(self, negative_field: float) -> None:
        if not isinstance(negative_field, (float, int)):
            raise e.TypeError("`negative_field` should be a float or integer")
        if negative_field < 0 or negative_field > 1:
            raise e.ValueError("`negative_field` should be between 0 and 1")
        if negative_field + self.positive_field > 1:
            raise e.ValueError(
                "`negative_field` + `positive_field` should not exceed 1"
            )

        self._negative_field = negative_field

    @property
    def ps_ratio(self) -> float:
        """Probability of selecting eletromagnets."""

        return self._ps_ratio

    @ps_ratio.setter
    def ps_ratio(self, ps_ratio: float) -> None:
        if not isinstance(ps_ratio, (float, int)):
            raise e.TypeError("`ps_ratio` should be a float or integer")
        if ps_ratio < 0 or ps_ratio > 1:
            raise e.ValueError("`ps_ratio` should be between 0 and 1")

        self._ps_ratio = ps_ratio

    @property
    def r_ratio(self) -> float:
        """Probability of selecting a random eletromagnet."""

        return self._r_ratio

    @r_ratio.setter
    def r_ratio(self, r_ratio: float) -> None:
        if not isinstance(r_ratio, (float, int)):
            raise e.TypeError("`r_ratio` should be a float or integer")
        if r_ratio < 0 or r_ratio > 1:
            raise e.ValueError("`r_ratio` should be between 0 and 1")

        self._r_ratio = r_ratio

    @property
    def phi(self) -> float:
        """Golden ratio."""

        return self._phi

    @phi.setter
    def phi(self, phi: float) -> None:
        if not isinstance(phi, (float, int)):
            raise e.TypeError("`phi` should be a float or integer")

        self._phi = phi

    @property
    def RI(self) -> float:
        """Eletromagnetic index."""

        return self._RI

    @RI.setter
    def RI(self, RI: float) -> None:
        if not isinstance(RI, int):
            raise e.TypeError("`RI` should be an integer")
        if RI < 0:
            raise e.TypeError("`RI` should be >= 0")

        self._RI = RI

    def _calculate_indexes(self, n_agents: int) -> Tuple[int, int, int]:
        """Calculates the indexes of positive, negative and neutral particles.

        Args:
            n_agents: Number of agents in the space.

        Returns:
            (Tuple[int, int, int]): Positive, negative and neutral particles' indexes.

        """

        # Calculates a positive particle's index
        positive_index = int(
            r.generate_uniform_random_number(0, n_agents * self.positive_field)
        )

        # Calculates a negative particle's index
        negative_index = int(
            r.generate_uniform_random_number(
                n_agents * (1 - self.negative_field), n_agents
            )
        )

        # Calculates a neutral particle's index
        neutral_index = int(
            r.generate_uniform_random_number(
                n_agents * self.positive_field, n_agents * (1 - self.negative_field)
            )
        )

        return positive_index, negative_index, neutral_index

    def update(self, space: Space, function: Function) -> None:
        """Wraps Electromagnetic Field Optimization over all agents and variables (eq. 1-4).

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        # Sorts agents according to their fitness
        space.agents.sort(key=lambda x: x.fit)

        # Gathers the number of total agents
        n_agents = len(space.agents)

        # Making a deepcopy of current's best agent
        agent = copy.deepcopy(space.agents[0])

        # Generates a uniform random number for the force
        force = r.generate_uniform_random_number()

        # For every decision variable
        for j in range(agent.n_variables):
            # Calculates the index of positive, negative and neutral particles
            pos, neg, neu = self._calculate_indexes(n_agents)

            # Generates a uniform random number
            r1 = r.generate_uniform_random_number()

            # If random number is smaller than the probability of selecting eletromagnets
            if r1 < self.ps_ratio:
                # Applies agent's position as positive particle's position
                agent.position[j] = space.agents[pos].position[j]

            # If random number is bigger
            else:
                # Calculates the new agent's position
                agent.position[j] = (
                    space.agents[neg].position[j]
                    + self.phi
                    * force
                    * (space.agents[pos].position[j] - space.agents[neu].position[j])
                    - force
                    * (space.agents[neg].position[j] - space.agents[neu].position[j])
                )

        # Clips the agent's position to its limits
        agent.clip_by_bound()

        # Generates a third uniform random number
        r2 = r.generate_uniform_random_number()

        # If random number is smaller than probability of changing a random eletromagnet
        if r2 < self.r_ratio:
            # Update agent's position based on RI
            agent.position[self.RI] = r.generate_uniform_random_number(
                agent.lb[self.RI], agent.ub[self.RI]
            )

            # Increases RI by one
            self.RI += 1

            # If RI exceeds the number of variables
            if self.RI >= agent.n_variables:
                # Resets it to one
                self.RI = 1

        # Calculates the agent's fitness
        agent.fit = function(agent.position)

        # If newly generated agent fitness is better than worst fitness
        if agent.fit < space.agents[-1].fit:
            # Updates the corresponding agent's object
            space.agents[-1] = copy.deepcopy(agent)

"""Univariate Marginal Distribution Algorithm.
"""

from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class UMDA(Optimizer):
    """An UMDA class, inherited from Optimizer.

    This is the designed class to define UMDA-related variables and methods.

    References:
        H. Mühlenbein. The equation for response to selection and its use for prediction.
        Evolutionary Computation (1997).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        # Overrides its parent class with the receiving params
        super(UMDA, self).__init__()

        # Probability of selection
        self.p_selection = 0.75

        # Distribution lower bound
        self.lower_bound = 0.05

        # Distribution upper bound
        self.upper_bound = 0.95

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def p_selection(self) -> float:
        """Probability of selection."""

        return self._p_selection

    @p_selection.setter
    def p_selection(self, p_selection: float) -> None:
        if not isinstance(p_selection, (float, int)):
            raise e.TypeError("`p_selection` should be a float or integer")
        if p_selection < 0 or p_selection > 1:
            raise e.ValueError("`p_selection` should be between 0 and 1")

        self._p_selection = p_selection

    @property
    def lower_bound(self) -> float:
        """Distribution lower bound."""

        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, lower_bound: float) -> None:
        if not isinstance(lower_bound, (float, int)):
            raise e.TypeError("`lower_bound` should be a float or integer")
        if lower_bound < 0 or lower_bound > 1:
            raise e.ValueError("`lower_bound` should be between 0 and 1")

        self._lower_bound = lower_bound

    @property
    def upper_bound(self) -> float:
        """Distribution upper bound."""

        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, upper_bound: float) -> None:
        if not isinstance(upper_bound, (float, int)):
            raise e.TypeError("`upper_bound` should be a float or integer")
        if upper_bound < 0 or upper_bound > 1:
            raise e.ValueError("`upper_bound` should be between 0 and 1")
        if upper_bound < self.lower_bound:
            raise e.ValueError("`upper_bound` should be greater than `lower_bound")

        self._upper_bound = upper_bound

    def _calculate_probability(self, agents: List[Agent]) -> np.ndarray:
        """Calculates probabilities based on pre-selected agents' variables occurrence (eq. 47).

        Args:
            agents: List of pre-selected agents.

        Returns:
            (np.ndarray): Probability of variables occurence.

        """

        # Creates an empty array of probabilities
        probs = np.zeros((agents[0].n_variables, agents[0].n_dimensions))

        # For every pre-selected agent
        for agent in agents:
            # Increases if feature is selected
            probs += agent.position

        # Normalizes into real probabilities
        probs /= len(agents)

        # Clips between pre-defined lower and upper bounds
        probs = np.clip(probs, self.lower_bound, self.upper_bound)

        return probs

    def _sample_position(self, probs: np.ndarray) -> np.ndarray:
        """Samples new positions according to their probability of ocurrence (eq. 53).

        Args:
            probs: Array of probabilities.

        Returns:
            (np.ndarray): New sampled position.

        """

        # Creates a uniform random array with the same shape as `probs`
        r1 = r.generate_uniform_random_number(size=(probs.shape[0], probs.shape[1]))

        # Samples new positions
        new_position = np.where(probs < r1, True, False)

        return new_position

    def update(self, space: Space) -> None:
        """Wraps Univariate Marginal Distribution Algorithm over all agents and variables.

        Args:
            space: Space containing agents and update-related information.

        """
        # Retrieves the number of agents
        n_agents = len(space.agents)

        # Selects the individuals through ranking
        n_selected = int(n_agents * self.p_selection)

        # Sorts agents
        space.agents.sort(key=lambda x: x.fit)

        # Calculates the probability of ocurrence from selected agents
        probs = self._calculate_probability(space.agents[:n_selected])

        # Iterates through every agents
        for agent in space.agents:
            # Samples new agent's position
            agent.position = self._sample_position(probs)

            # Checks its limits
            agent.clip_by_bound()

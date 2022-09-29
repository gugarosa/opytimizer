"""Cross-Entropy Method.
"""

from typing import Any, Dict, List, Optional

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.function import Function
from opytimizer.core.space import Space
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class CEM(Optimizer):
    """A CEM class, inherited from Optimizer.

    This is the designed class to define CEM-related
    variables and methods.

    References:
        R. Y. Rubinstein. Optimization of Computer simulation Models with Rare Events.
        European Journal of Operations Research (1997).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        super(CEM, self).__init__()

        self.n_updates = 5
        self.alpha = 0.7

        self.build(params)

        logger.info("Class overrided.")

    @property
    def n_updates(self) -> int:
        """Number of positions to employ in update formulae."""

        return self._n_updates

    @n_updates.setter
    def n_updates(self, n_updates: int) -> None:
        if not isinstance(n_updates, int):
            raise e.TypeError("`n_updates` should be an integer")
        if n_updates <= 0:
            raise e.ValueError("`n_updates` should be > 0")

        self._n_updates = n_updates

    @property
    def alpha(self) -> float:
        """Learning rate."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if not isinstance(alpha, (float, int)):
            raise e.TypeError("`alpha` should be a float or integer")
        if alpha < 0:
            raise e.ValueError("`alpha` should be >= 0")

        self._alpha = alpha

    @property
    def mean(self) -> np.ndarray:
        """Array of means."""

        return self._mean

    @mean.setter
    def mean(self, mean: np.ndarray) -> None:
        if not isinstance(mean, np.ndarray):
            raise e.TypeError("`mean` should be a numpy array")

        self._mean = mean

    @property
    def std(self) -> np.ndarray:
        """Array of standard deviations."""

        return self._std

    @std.setter
    def std(self, std: np.ndarray) -> None:
        if not isinstance(std, np.ndarray):
            raise e.TypeError("`std` should be a numpy array")

        self._std = std

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.mean = np.zeros(space.n_variables)
        self.std = np.zeros(space.n_variables)

        for j, (lb, ub) in enumerate(zip(space.lb, space.ub)):
            self.mean[j] = r.generate_uniform_random_number(lb, ub)
            self.std[j] = ub - lb

    def _create_new_samples(self, agents: List[Agent], function: Function) -> None:
        """Creates new agents based on current mean and standard deviation.

        Args:
            agents (list): List of agents.
            function: A Function object that will be used as the objective function.

        """

        for agent in agents:
            for j, (m, s) in enumerate(zip(self.mean, self.std)):
                agent.position[j] = r.generate_gaussian_random_number(
                    m, s, agent.n_dimensions
                )

            agent.clip_by_bound()

            agent.fit = function(agent.position)

    def _update_mean(self, updates: np.ndarray) -> np.ndarray:
        """Calculates and updates mean.

        Args:
            updates: An array of updates' positions.

        Returns:
            (np.ndarray): The new mean values.

        """

        new_mean = self.alpha * self.mean + (1 - self.alpha) * np.mean(updates)

        return new_mean

    def _update_std(self, updates: np.ndarray) -> np.ndarray:
        """Calculates and updates standard deviation.

        Args:
            updates: An array of updates' positions.

        Returns:
            (np.ndarray): The new standard deviation values.

        """

        new_std = self.alpha * self.std + (1 - self.alpha) * np.sqrt(
            np.mean((updates - self.mean) ** 2)
        )

        return new_std

    def update(self, space: Space, function: Function) -> None:
        """Wraps Cross-Entropy Method over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A Function object that will be used as the objective function.

        """

        self._create_new_samples(space.agents, function)

        space.agents.sort(key=lambda x: x.fit)

        update_position = np.array(
            [agent.position for agent in space.agents[: self.n_updates]]
        )

        self.mean = self._update_mean(update_position)
        self.std = self._update_std(update_position)

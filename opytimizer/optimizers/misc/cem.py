"""Cross-Entropy Method."""

from collections.abc import Callable, Mapping
from numbers import Real
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class CEM(Optimizer):
    """Fit per-variable Gaussian sampling distributions to elite candidates.

    Attributes:
        n_updates: Positive number of best agents used to update the distribution.
            Defaults to ``5``; if larger than the population, all agents are used.
        alpha: Non-negative weight of the previous mean and standard deviation.
            Defaults to ``0.7``. Values above one are accepted but extrapolate
            rather than form a convex average.
        mean: Sampling means of shape ``(n_variables,)``, created by ``compile``.
        std: Sampling standard deviations of shape ``(n_variables,)``, initially
            set to each variable's bound span.

    Each variable shares its distribution across dimensions. Updates first
    smooth the mean, then calculate deviations around that updated mean.

    References:
        R. Y. Rubinstein. Optimization of Computer simulation Models with Rare Events.
        European Journal of Operations Research (1997).

    """

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        """Configure elite selection and distribution smoothing.

        Args:
            params: Optional overrides for ``n_updates`` and ``alpha``.

        """

        super().__init__()

        self.n_updates = 5
        self.alpha = 0.7

        self.build(params)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if not isinstance(self.n_updates, (int, np.integer)):
            raise TypeError("`n_updates` should be an integer")
        if self.n_updates <= 0:
            raise ValueError("`n_updates` should be > 0")
        if not isinstance(self.alpha, Real):
            raise TypeError("`alpha` should be a real number")
        if not self.alpha >= 0:
            raise ValueError("`alpha` should be >= 0")

    def compile(self, space: Space) -> None:
        """Compiles additional information that is used by this optimizer.

        Args:
            space: A Space object containing meta-information.

        """

        self.mean = np.zeros(space.n_variables)
        self.std = np.zeros(space.n_variables)

        for j, (lb, ub) in enumerate(zip(space.lb, space.ub)):
            self.mean[j] = np.random.uniform(lb, ub)
            self.std[j] = ub - lb

    def _create_new_samples(self, agents: list[Agent], function: Callable) -> None:
        """Creates new agents based on current mean and standard deviation.

        Args:
            agents (list): List of agents.
            function: A callable that will be used as the objective function.

        """

        for agent in agents:
            for j, (m, s) in enumerate(zip(self.mean, self.std)):
                agent.position[j] = np.random.normal(m, s, agent.n_dimensions)

            agent.clip_by_bound()

            agent.fit = function(agent.position)

    def _update_mean(self, updates: np.ndarray) -> np.ndarray:
        """Return smoothed means without mutating the current mean buffer.

        Args:
            updates: Elite positions of shape
                ``(n_elites, n_variables, n_dimensions)``.

        Returns:
            (np.ndarray): The new mean values.

        """

        new_mean = self.alpha * self.mean + (1 - self.alpha) * np.mean(
            updates, axis=(0, 2)
        )

        return new_mean

    def _update_std(self, updates: np.ndarray) -> np.ndarray:
        """Return smoothed deviations around the current per-variable means.

        Args:
            updates: Elite positions of shape
                ``(n_elites, n_variables, n_dimensions)``.

        Returns:
            (np.ndarray): The new standard deviation values.

        """

        new_std = self.alpha * self.std + (1 - self.alpha) * np.sqrt(
            np.mean((updates - self.mean[None, :, None]) ** 2, axis=(0, 2))
        )

        return new_std

    def update(self, space: Space, function: Callable) -> None:
        """Wraps Cross-Entropy Method over all agents and variables.

        Args:
            space: Space containing agents and update-related information.
            function: A callable that will be used as the objective function.

        """

        self._validate_parameters()
        self._create_new_samples(space.agents, function)

        space.agents.sort(key=lambda x: x.fit)

        update_position = np.array(
            [agent.position for agent in space.agents[: self.n_updates]]
        )

        self.mean = self._update_mean(update_position)
        self.std = self._update_std(update_position)

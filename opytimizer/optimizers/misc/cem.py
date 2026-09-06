# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Cross-Entropy Method.

Compilation creates ``mean`` and ``std`` arrays with shape ``(n_variables,)``, initializing
means uniformly within bounds and standard deviations to bound spans. Each variable shares
its distribution across dimensions. Updates smooth the mean before calculating deviations
around that updated mean from elite positions of shape ``(n_elites, n_variables, n_dimensions)``.
The mean and deviation helpers return new arrays without mutating their input buffers.

References:
    R. Y. Rubinstein. Optimization of Computer simulation Models with Rare Events.
    European Journal of Operations Research (1997).

"""

from collections.abc import Callable, Mapping
from numbers import Real
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class CEM(Optimizer):
    """Fit per-variable Gaussian sampling distributions to elite candidates.

    """

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        """Configure elite selection and distribution smoothing.

        Args:
            params: Optional overrides for ``n_updates`` and ``alpha``.

        Notes:
            Supported keys are ``n_updates`` (positive elite count, 5) and ``alpha``
            (nonnegative weight of previous distribution parameters, 0.7).
            Elite counts above the population size use all agents. Weights above one
            are accepted but extrapolate rather than form a convex average.

        Raises:
            TypeError: The elite count is not integral or the smoothing weight is not real.
            ValueError: The elite count is not positive or the smoothing weight is negative.

        """

        super().__init__()

        self.n_updates = 5
        self.alpha = 0.7

        self.build(params)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if not isinstance(self.n_updates, (int, np.integer)):
            raise TypeError("`n_updates` should be an integer.")
        if self.n_updates <= 0:
            raise ValueError("`n_updates` should be > 0.")
        if not isinstance(self.alpha, Real):
            raise TypeError("`alpha` should be a real number.")
        if not self.alpha >= 0:
            raise ValueError("`alpha` should be >= 0.")

    def compile(self, space: Space) -> None:
        self.mean = np.zeros(space.n_variables)
        self.std = np.zeros(space.n_variables)

        for j, (lb, ub) in enumerate(zip(space.lb, space.ub)):
            self.mean[j] = np.random.uniform(lb, ub)
            self.std[j] = ub - lb

    def _create_new_samples(self, agents: list[Agent], function: Callable) -> None:
        for agent in agents:
            for j, (m, s) in enumerate(zip(self.mean, self.std)):
                agent.position[j] = np.random.normal(m, s, agent.n_dimensions)

            agent.clip_by_bound()

            agent.fit = function(agent.position)

    def _update_mean(self, updates: np.ndarray) -> np.ndarray:
        new_mean = self.alpha * self.mean + (1 - self.alpha) * np.mean(updates, axis=(0, 2))

        return new_mean

    def _update_std(self, updates: np.ndarray) -> np.ndarray:
        new_std = self.alpha * self.std + (1 - self.alpha) * np.sqrt(
            np.mean((updates - self.mean[None, :, None]) ** 2, axis=(0, 2))
        )

        return new_std

    def update(self, space: Space, function: Callable) -> None:
        self._validate_parameters()
        self._create_new_samples(space.agents, function)

        space.agents.sort(key=lambda x: x.fit)

        update_position = np.array([agent.position for agent in space.agents[: self.n_updates]])

        self.mean = self._update_mean(update_position)
        self.std = self._update_std(update_position)

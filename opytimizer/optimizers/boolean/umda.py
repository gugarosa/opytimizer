# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Univariate Marginal Distribution Algorithm.

Marginal frequencies are clipped using equation 47 and resampled using equation 53.
Sampling retains the comparison ``probs < uniform_draw`` used by this implementation.
Bounds are validated at construction and before updates or direct probability calculation.

References:
    H. Mühlenbein. The equation for response to selection and its use for prediction.
    Evolutionary Computation (1997).

"""

from numbers import Real
from typing import Any

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space


class UMDA(Optimizer):
    """Optimize Boolean variables using univariate marginal distributions.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the selection fraction and probability bounds.

        Args:
            params: Overrides for the supported optimizer parameters.

        Notes:
            Supported keys are ``p_selection`` (selected population fraction, 0.75),
            ``lower_bound`` (minimum marginal probability, 0.05), and
            ``upper_bound`` (maximum marginal probability, 0.95).
            Bounds must satisfy ``0 <= lower_bound <= upper_bound <= 1``.

        Raises:
            TypeError: A probability bound is not a real number.
            ValueError: Probability bounds are outside their ordered unit interval.

        """

        super(UMDA, self).__init__()

        self.p_selection = 0.75
        self.lower_bound = 0.05
        self.upper_bound = 0.95

        self.build(params)
        self._validate_probability_bounds()

    def _validate_probability_bounds(self) -> None:
        for name in ("lower_bound", "upper_bound"):
            if not isinstance(getattr(self, name), Real):
                raise TypeError(f"`{name}` must be a real number.")
        if not 0 <= self.lower_bound <= self.upper_bound <= 1:
            raise ValueError("`lower_bound` and `upper_bound` must satisfy 0 <= lower_bound <= upper_bound <= 1.")

    def _calculate_probability(self, agents: list[Agent]) -> np.ndarray:
        self._validate_probability_bounds()

        probs = np.zeros((agents[0].n_variables, agents[0].n_dimensions))

        for agent in agents:
            probs += agent.position

        probs /= len(agents)
        probs = np.clip(probs, self.lower_bound, self.upper_bound)

        return probs

    def _sample_position(self, probs: np.ndarray) -> np.ndarray:
        r1 = np.random.uniform(0.0, 1.0, (probs.shape[0], probs.shape[1]))

        new_position = np.where(probs < r1, True, False)

        return new_position

    def update(self, space: Space) -> None:
        self._validate_probability_bounds()

        n_agents = len(space.agents)
        n_selected = int(n_agents * self.p_selection)

        space.agents.sort(key=lambda x: x.fit)

        probs = self._calculate_probability(space.agents[:n_selected])

        for agent in space.agents:
            agent.position = self._sample_position(probs)
            agent.clip_by_bound()

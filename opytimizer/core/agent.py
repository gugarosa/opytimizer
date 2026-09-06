# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Agent.

"""

import time

import numpy as np
from numpy.typing import ArrayLike

import opytimizer.utils.constant as c


class Agent:
    """Hold one mutable candidate position and optimizer-maintained metadata.

    """

    def __init__(
        self,
        n_variables: int,
        n_dimensions: int,
        lower_bound: ArrayLike | None,
        upper_bound: ArrayLike | None,
        mapping: list[str] | None = None,
    ) -> None:
        """Allocate a zero position and retain the supplied bound configuration.

        Args:
            n_variables: Number of decision variables.
            n_dimensions: Number of dimensions.
            lower_bound: Per-variable lower bounds. A scalar is valid for one variable.
            upper_bound: Per-variable upper bounds. A scalar is valid for one variable.
            mapping: String-based identifiers for mapping variables' names.

        Raises:
            TypeError: Dimensions are not integers or a supplied mapping is not a list.
            ValueError: Dimensions are not positive or bounds and mapping do not match the variable count.

        Notes:
            ``position`` starts as zeros with shape ``(n_variables, n_dimensions)``.
            Optimizers own ``fit`` and may use it for a personal-best score rather
            than the current position's score. ``lb`` and ``ub`` retain array
            inputs without copying, and ``mapping`` retains the supplied list.
            ``ts`` is a Unix creation or best-improvement timestamp, not a duration.
            ``None`` bounds remain available for base-space configuration but do
            not provide numeric limits for clipping or uniform initialization.

        """

        if not isinstance(n_variables, int):
            raise TypeError("`n_variables` should be an integer.")
        if n_variables <= 0:
            raise ValueError("`n_variables` should be > 0.")
        if not isinstance(n_dimensions, int):
            raise TypeError("`n_dimensions` should be an integer.")
        if n_dimensions <= 0:
            raise ValueError("`n_dimensions` should be > 0.")

        lb = np.asarray(lower_bound)
        ub = np.asarray(upper_bound)
        if not lb.shape:
            lb = np.expand_dims(lb, -1)
        if not ub.shape:
            ub = np.expand_dims(ub, -1)
        if lb.shape[0] != n_variables:
            raise ValueError("`lower_bound` should match `n_variables`.")
        if ub.shape[0] != n_variables:
            raise ValueError("`upper_bound` should match `n_variables`.")

        if mapping is None:
            mapping = [f"x{i}" for i in range(n_variables)]
        elif not isinstance(mapping, list):
            raise TypeError("`mapping` should be a list.")
        elif len(mapping) != n_variables:
            raise ValueError("`mapping` should match `n_variables`.")

        self.n_variables = n_variables
        self.n_dimensions = n_dimensions

        self.position = np.zeros((n_variables, n_dimensions))
        self.fit = c.FLOAT_MAX

        self.lb = lb
        self.ub = ub
        self.mapping = mapping

        self.ts = int(time.time())

    @property
    def mapped_position(self) -> dict[str, np.ndarray]:
        """Map variable names to live position rows, not independent copies.

        """

        return dict(zip(self.mapping, self.position))

    def clip_by_bound(self) -> None:
        """Clips the agent's decision variables to the bounds limits.

        """

        for j, (lb, ub) in enumerate(zip(self.lb, self.ub)):
            self.position[j] = np.clip(self.position[j], lb, ub)

    def fill_with_binary(self) -> None:
        """Fills the agent's decision variables with a binary distribution.

        """

        for j in range(self.n_variables):
            self.position[j] = np.round(np.random.uniform(0, 1, self.n_dimensions))

    def fill_with_static(self, values: ArrayLike) -> None:
        """Fill positions without enforcing bounds.

        Per-variable values broadcast across dimensions. A matrix must match
        ``position``, and a scalar is accepted for one variable.

        Args:
            values: Scalar, sequence, or array of values copied into the position.

        Raises:
            ValueError: Values have the wrong variable count or cannot broadcast into the position rows.

        """

        values = np.asarray(values)
        if not values.shape:
            values = np.expand_dims(values, -1)
        if values.shape[0] != self.n_variables:
            raise ValueError("`values` should match `n_variables`.")

        for j, value in enumerate(values):
            self.position[j] = value

    def fill_with_uniform(self) -> None:
        """Fill the position in place with uniform samples within each variable's bounds.

        """

        for j, (lb, ub) in enumerate(zip(self.lb, self.ub)):
            self.position[j] = np.random.uniform(lb, ub, self.n_dimensions)

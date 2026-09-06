"""Agent."""

import time
from typing import Dict, List, Optional, Union

import numpy as np

import opytimizer.utils.constant as c


class Agent:
    """Hold one mutable candidate position and optimizer-maintained metadata.

    Attributes:
        position: Array of shape ``(n_variables, n_dimensions)``, initially zero.
        fit: Fitness used by the optimizer. Some strategies, such as PSO, keep a
            personal-best score here rather than the current position's score.
        lb: Per-variable lower bounds.
        ub: Per-variable upper bounds.
        mapping: Variable names used by ``mapped_position``.
        ts: Unix timestamp used for creation or best-improvement metadata, not
            an elapsed-duration clock.
    """

    def __init__(
        self,
        n_variables: int,
        n_dimensions: int,
        lower_bound: List[Union[int, float]],
        upper_bound: List[Union[int, float]],
        mapping: Optional[List[str]] = None,
    ) -> None:
        """Allocate a zero position and retain the supplied bound configuration.

        Args:
            n_variables: Number of decision variables.
            n_dimensions: Number of dimensions.
            lower_bound: Per-variable lower bounds. A scalar is valid for one variable.
            upper_bound: Per-variable upper bounds. A scalar is valid for one variable.
            mapping: String-based identifiers for mapping variables' names.

        """

        if not isinstance(n_variables, int):
            raise TypeError("`n_variables` should be an integer")
        if n_variables <= 0:
            raise ValueError("`n_variables` should be > 0")
        if not isinstance(n_dimensions, int):
            raise TypeError("`n_dimensions` should be an integer")
        if n_dimensions <= 0:
            raise ValueError("`n_dimensions` should be > 0")

        lb = np.asarray(lower_bound)
        ub = np.asarray(upper_bound)
        if not lb.shape:
            lb = np.expand_dims(lb, -1)
        if not ub.shape:
            ub = np.expand_dims(ub, -1)
        if lb.shape[0] != n_variables:
            raise ValueError("`lower_bound` should match `n_variables`")
        if ub.shape[0] != n_variables:
            raise ValueError("`upper_bound` should match `n_variables`")

        if mapping is None:
            mapping = [f"x{i}" for i in range(n_variables)]
        elif not isinstance(mapping, list):
            raise TypeError("`mapping` should be a list")
        elif len(mapping) != n_variables:
            raise ValueError("`mapping` should match `n_variables`")

        self.n_variables = n_variables
        self.n_dimensions = n_dimensions

        self.position = np.zeros((n_variables, n_dimensions))
        self.fit = c.FLOAT_MAX

        self.lb = lb
        self.ub = ub
        self.mapping = mapping

        self.ts = int(time.time())

    @property
    def mapped_position(self) -> Dict[str, np.ndarray]:
        """Map variable names to live position rows, not independent copies."""

        return dict(zip(self.mapping, self.position))

    def clip_by_bound(self) -> None:
        """Clips the agent's decision variables to the bounds limits."""

        for j, (lb, ub) in enumerate(zip(self.lb, self.ub)):
            self.position[j] = np.clip(self.position[j], lb, ub)

    def fill_with_binary(self) -> None:
        """Fills the agent's decision variables with a binary distribution."""

        for j in range(self.n_variables):
            self.position[j] = np.round(np.random.uniform(0, 1, self.n_dimensions))

    def fill_with_static(self, values: np.ndarray) -> None:
        """Fill positions without enforcing bounds.

        Args:
            values: Per-variable values, broadcast across dimensions, or a matrix
                matching ``position``. A scalar is accepted for one variable.
        """

        values = np.asarray(values)
        if not values.shape:
            values = np.expand_dims(values, -1)
        if values.shape[0] != self.n_variables:
            raise ValueError("`values` should match `n_variables`")

        for j, value in enumerate(values):
            self.position[j] = value

    def fill_with_uniform(self) -> None:
        """Fills the agent's decision variables with a uniform distribution
        based on bounds limits.

        """

        for j, (lb, ub) in enumerate(zip(self.lb, self.ub)):
            self.position[j] = np.random.uniform(lb, ub, self.n_dimensions)

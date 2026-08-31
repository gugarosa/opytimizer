"""Agent."""

import time
from typing import Dict, List, Optional, Union

import numpy as np

import opytimizer.utils.constant as c


class Agent:
    """An Agent class for all optimization techniques."""

    def __init__(
        self,
        n_variables: int,
        n_dimensions: int,
        lower_bound: List[Union[int, float]],
        upper_bound: List[Union[int, float]],
        mapping: Optional[List[str]] = None,
    ) -> None:
        """Initialization method.

        Args:
            n_variables: Number of decision variables.
            n_dimensions: Number of dimensions.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
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
        """Dictionary mapping variables names and array of positions."""

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
        """Fills the agent's decision variables with static values. Note that this
        method ignore the agent's bounds, so use it carefully.

        Args:
            values: Values to be filled.

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

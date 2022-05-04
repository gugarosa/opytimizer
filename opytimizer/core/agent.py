"""Agent.
"""

import time
from typing import List, Union

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class Agent:
    """An Agent class for all optimization techniques."""

    def __init__(
        self,
        n_variables: int,
        n_dimensions: int,
        lower_bound: List[Union[int, float]],
        upper_bound: List[Union[int, float]],
    ) -> None:
        """Initialization method.

        Args:
            n_variables: Number of decision variables.
            n_dimensions: Number of dimensions.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.

        """

        # Number of decision variables
        self.n_variables = n_variables

        # Number of dimensions
        self.n_dimensions = n_dimensions

        # N-dimensional array of positions
        self.position = np.zeros((n_variables, n_dimensions))

        # Fitness value (largest float number)
        self.fit = c.FLOAT_MAX

        # Lower bounds
        self.lb = np.asarray(lower_bound)

        # Upper bounds
        self.ub = np.asarray(upper_bound)

        # Timestamp
        self.ts = int(time.time())

    @property
    def n_variables(self) -> int:
        """Number of decision variables."""

        return self._n_variables

    @n_variables.setter
    def n_variables(self, n_variables: int) -> None:
        if not isinstance(n_variables, int):
            raise e.TypeError("`n_variables` should be an integer")
        if n_variables <= 0:
            raise e.ValueError("`n_variables` should be > 0")

        self._n_variables = n_variables

    @property
    def n_dimensions(self) -> int:
        """Number of dimensions."""

        return self._n_dimensions

    @n_dimensions.setter
    def n_dimensions(self, n_dimensions: int) -> None:
        if not isinstance(n_dimensions, int):
            raise e.TypeError("`n_dimensions` should be an integer")
        if n_dimensions <= 0:
            raise e.ValueError("`n_dimensions` should be > 0")

        self._n_dimensions = n_dimensions

    @property
    def position(self) -> np.ndarray:
        """N-dimensional array of positions."""

        return self._position

    @position.setter
    def position(self, position: np.ndarray) -> None:
        if not isinstance(position, np.ndarray):
            raise e.TypeError("`position` should be a numpy array")

        self._position = position

    @property
    def fit(self) -> Union[int, float]:
        """float: Fitness value."""

        return self._fit

    @fit.setter
    def fit(self, fit: Union[int, float]) -> None:
        if not isinstance(fit, (float, int, np.int32, np.int64)):
            raise e.TypeError("`fit` should be a float or integer")

        self._fit = fit

    @property
    def lb(self) -> np.ndarray:
        """Lower bounds."""

        return self._lb

    @lb.setter
    def lb(self, lb: np.ndarray) -> None:
        if not isinstance(lb, np.ndarray):
            raise e.TypeError("`lb` should be a numpy array")
        if not lb.shape:
            lb = np.expand_dims(lb, -1)
        if lb.shape[0] != self.n_variables:
            raise e.SizeError("`lb` should be the same size as `n_variables`")

        self._lb = lb

    @property
    def ub(self) -> np.ndarray:
        """Upper bounds."""

        return self._ub

    @ub.setter
    def ub(self, ub: np.ndarray) -> None:
        if not isinstance(ub, np.ndarray):
            raise e.TypeError("`ub` should be a numpy array")
        if not ub.shape:
            ub = np.expand_dims(ub, -1)
        if ub.shape[0] != self.n_variables:
            raise e.SizeError("`ub` should be the same size as `n_variables`")

        self._ub = ub

    @property
    def ts(self) -> int:
        """Timestamp of the agent."""

        return self._ts

    @ts.setter
    def ts(self, ts: int) -> None:
        if not isinstance(ts, int):
            raise e.TypeError("`ts` should be an integer")

        self._ts = ts

    def clip_by_bound(self) -> None:
        """Clips the agent's decision variables to the bounds limits."""

        # Iterates through all the decision variables
        for j, (lb, ub) in enumerate(zip(self.lb, self.ub)):
            # Clips the array based on variable's lower and upper bounds
            self.position[j] = np.clip(self.position[j], lb, ub)

    def fill_with_binary(self) -> None:
        """Fills the agent's decision variables with a binary distribution."""

        # Iterates through all the decision variables
        for j in range(self.n_variables):
            # Fills the array based on a binary distribution
            self.position[j] = r.generate_binary_random_number(self.n_dimensions)

    def fill_with_static(self, values: np.ndarray) -> None:
        """Fills the agent's decision variables with static values. Note that this
        method ignore the agent's bounds, so use it carefully.

        Args:
            values: Values to be filled.

        """

        # Makes sure that `values` is a numpy array
        # and has the same size of `n_variables`
        values = np.asarray(values)
        if not values.shape:
            values = np.expand_dims(values, -1)
        if values.shape[0] != self.n_variables:
            raise e.SizeError("`values` should be the same size as `n_variables`")

        # Iterates through all the decision variables
        for j, value in enumerate(values):
            # Fills the array based on a static value
            self.position[j] = value

    def fill_with_uniform(self) -> None:
        """Fills the agent's decision variables with a uniform distribution
        based on bounds limits.

        """

        # Iterates through all the decision variables
        for j, (lb, ub) in enumerate(zip(self.lb, self.ub)):
            # Fills the array based on a uniform distribution
            self.position[j] = r.generate_uniform_random_number(
                lb, ub, self.n_dimensions
            )

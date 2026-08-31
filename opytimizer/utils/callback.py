"""Callbacks."""

from typing import List, TypeVar, Union

import numpy as np

from opytimizer.core.space import Space

Opytimizer = TypeVar("Opytimizer")


class Callback:
    """A Callback class that handles additional variables and methods
    manipulation that are not provided by the library.

    """

    def on_task_begin(self, opt_model: Opytimizer) -> None:
        """Performs a callback whenever a task begins.

        Args:
            opt_model: An instance of the optimization model.

        """

        pass

    def on_task_end(self, opt_model: Opytimizer) -> None:
        """Performs a callback whenever a task ends.

        Args:
            opt_model: An instance of the optimization model.

        """

        pass

    def on_iteration_begin(self, iteration: int, opt_model: Opytimizer) -> None:
        """Performs a callback whenever an iteration begins.

        Args:
            iteration: Current iteration.
            opt_model: An instance of the optimization model.

        """

        pass

    def on_iteration_end(self, iteration: int, opt_model: Opytimizer) -> None:
        """Performs a callback whenever an iteration ends.

        Args:
            iteration: Current iteration.
            opt_model: An instance of the optimization model.

        """

        pass

    def on_evaluate_before(self, *evaluate_args) -> None:
        """Performs a callback prior to the `evaluate` method."""

        pass

    def on_evaluate_after(self, *evaluate_args) -> None:
        """Performs a callback after the `evaluate` method."""

        pass

    def on_update_before(self, *update_args) -> None:
        """Performs a callback prior to the `update` method."""

        pass

    def on_update_after(self, *update_args) -> None:
        """Performs a callback after the `update` method."""

        pass


class CheckpointCallback(Callback):
    """A callback that periodically saves the optimization model."""

    def __init__(self, file_path: str = None, frequency: int = 0) -> None:
        """Initialization method.

        Args:
            file_path: Path of file to be saved.
            frequency: Interval between checkpoints.

        """

        if file_path is None:
            file_path = "checkpoint.pkl"
        if not isinstance(file_path, str):
            raise TypeError("`file_path` should be a string")
        if not isinstance(frequency, int):
            raise TypeError("`frequency` should be an integer")
        if frequency < 0:
            raise ValueError("`frequency` should be >= 0")

        self.file_path = file_path
        self.frequency = frequency

    def on_iteration_end(self, iteration: int, opt_model: Opytimizer) -> None:
        """Performs a callback whenever an iteration ends.

        Args:
            iteration: Current iteration.
            opt_model: An instance of the optimization model.

        """

        if self.frequency > 0 and iteration % self.frequency == 0:
            opt_model.save(f"iter_{iteration}_{self.file_path}")


class DiscreteSearchCallback(Callback):
    """A DiscreteSearchCallback class that handles mapping floating-point variables
    to discrete values.

    """

    def __init__(self, allowed_values: List[Union[int, float]] = None) -> None:
        """Initialization method.

        Args:
            allowed_values: Possible values between lower and upper bounds that variables can be mapped.

        """

        if allowed_values is None:
            allowed_values = []
        if not isinstance(allowed_values, list):
            raise TypeError("`allowed_values` should be a list")

        self.allowed_values = allowed_values

    def on_task_begin(self, opt_model: Opytimizer) -> None:
        """Performs a callback whenever a task begins.

        Args:
            opt_model: An instance of the optimization model.

        """

        n_variables = opt_model.space.n_variables
        lower_bound = opt_model.space.lb
        upper_bound = opt_model.space.ub

        if len(self.allowed_values) != n_variables:
            raise ValueError(f"`allowed_values` should contain {n_variables} lists")
        if not all(
            np.all((np.asarray(values) >= lower) & (np.asarray(values) <= upper))
            for values, lower, upper in zip(
                self.allowed_values, lower_bound, upper_bound
            )
        ):
            raise ValueError("`allowed_values` should stay within the space bounds")

    def on_evaluate_before(self, *evaluate_args) -> None:
        """Performs a callback prior to the `evaluate` method."""

        space = evaluate_args[0]
        if not isinstance(space, Space):
            raise TypeError("the first evaluate argument should be a Space")

        for agent in space.agents:
            for i in range(agent.n_variables):
                min_value_idx = np.argmin(
                    abs(agent.position[i] - self.allowed_values[i])
                )
                agent.position[i] = self.allowed_values[i][min_value_idx]

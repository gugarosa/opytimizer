"""Callbacks."""

from __future__ import annotations

from os import PathLike, fspath
from pathlib import Path
from typing import Any

import numpy as np

import opytimizer
from opytimizer.core.space import Space


class Callback:
    """Observe or deliberately modify an optimization through lifecycle hooks.

    Override only the hooks you need; all defaults do nothing. Hooks execute in
    callback sequence order and receive live objects, not snapshots. Exceptions
    propagate to the caller, so task-end hooks are not a ``finally`` mechanism.
    """

    def on_task_begin(self, opt_model: opytimizer.Opytimizer) -> None:
        """Run after budget validation and before initial evaluation.

        Args:
            opt_model: An instance of the optimization model.

        """

        pass

    def on_task_end(self, opt_model: opytimizer.Opytimizer) -> None:
        """Run on normal completion, before this run's elapsed time is recorded.

        Args:
            opt_model: An instance of the optimization model.

        """

        pass

    def on_iteration_begin(
        self, iteration: int, opt_model: opytimizer.Opytimizer
    ) -> None:
        """Run before an update, using the cumulative one-based iteration counter.

        Args:
            iteration: Cumulative iteration counter, already incremented.
            opt_model: An instance of the optimization model.

        """

        pass

    def on_iteration_end(
        self, iteration: int, opt_model: opytimizer.Opytimizer
    ) -> None:
        """Run after evaluation and history recording for a completed iteration.

        Args:
            iteration: Cumulative iteration counter, not the zero-based run index.
            opt_model: An instance of the optimization model.

        """

        pass

    def on_evaluate_before(self, *evaluate_args: Any) -> None:
        """Receive the evaluator's resolved arguments before it runs."""

        pass

    def on_evaluate_after(self, *evaluate_args: Any) -> None:
        """Receive the evaluator's resolved arguments after it returns normally."""

        pass

    def on_update_before(self, *update_args: Any) -> None:
        """Receive the updater's resolved arguments before it runs."""

        pass

    def on_update_after(self, *update_args: Any) -> None:
        """Receive updater arguments after updating, before driver bound clipping."""

        pass


class CheckpointCallback(Callback):
    """Save completed iterations using the model's dill serialization path."""

    def __init__(
        self, file_path: str | PathLike[str] | None = None, frequency: int = 0
    ) -> None:
        """Configure a checkpoint filename and iteration interval.

        Args:
            file_path: Path of file to be saved. The iteration prefix is added to
                the filename, preserving its directory, which must already exist.
            frequency: Interval in cumulative iterations. Zero disables saving.

        """

        if file_path is None:
            file_path = "checkpoint.pkl"
        if isinstance(file_path, PathLike):
            file_path = fspath(file_path)
        if not isinstance(file_path, str):
            raise TypeError("`file_path` should be a string or text path-like object")
        if not isinstance(frequency, int):
            raise TypeError("`frequency` should be an integer")
        if frequency < 0:
            raise ValueError("`frequency` should be >= 0")

        self.file_path = file_path
        self.frequency = frequency

    def on_iteration_end(
        self, iteration: int, opt_model: opytimizer.Opytimizer
    ) -> None:
        """Save when the completed cumulative iteration reaches the interval.

        Args:
            iteration: Current iteration.
            opt_model: An instance of the optimization model.

        """

        if self.frequency > 0 and iteration % self.frequency == 0:
            path = Path(self.file_path)
            opt_model.save(str(path.with_name(f"iter_{iteration}_{path.name}")))


class DiscreteSearchCallback(Callback):
    """Project each position component onto a variable's allowed values."""

    def __init__(self, allowed_values: list[list[int | float]] | None = None) -> None:
        """Configure one ordered set of discrete candidates per variable.

        Args:
            allowed_values: One non-empty list of possible values per variable.
                Every dimension is mapped independently to its nearest value;
                ties select the first listed value.

        """

        if allowed_values is None:
            allowed_values = []
        if not isinstance(allowed_values, list):
            raise TypeError("`allowed_values` should be a list")

        self.allowed_values = allowed_values

    def on_task_begin(self, opt_model: opytimizer.Opytimizer) -> None:
        """Validate candidate sets against the current space before evaluation.

        Args:
            opt_model: An instance of the optimization model.

        """

        n_variables = opt_model.space.n_variables
        lower_bound = opt_model.space.lb
        upper_bound = opt_model.space.ub

        if len(self.allowed_values) != n_variables:
            raise ValueError(f"`allowed_values` should contain {n_variables} lists")
        for values, lower, upper in zip(self.allowed_values, lower_bound, upper_bound):
            values = np.asarray(values)
            if values.ndim != 1 or values.size == 0:
                raise ValueError("`allowed_values` should contain non-empty vectors")
            if not np.all((values >= lower) & (values <= upper)):
                raise ValueError("`allowed_values` should stay within the space bounds")

    def on_evaluate_before(self, *evaluate_args: Any) -> None:
        """Project live positions; the evaluator's first argument must be a space."""

        space = evaluate_args[0]
        if not isinstance(space, Space):
            raise TypeError("the first evaluate argument should be a Space")

        for agent in space.agents:
            for i in range(agent.n_variables):
                values = np.asarray(self.allowed_values[i])
                min_value_idx = np.argmin(
                    np.abs(agent.position[i, :, None] - values), axis=1
                )
                agent.position[i] = values[min_value_idx]

# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Provide optimization lifecycle hooks and reusable callbacks.

Hooks receive live objects and execute in callback order.
Override only the hooks required by the application. Errors propagate and task-end hooks are not cleanup guards.

"""

from __future__ import annotations

from os import PathLike, fspath
from pathlib import Path
from typing import Any

import numpy as np

import opytimizer
from opytimizer.core.space import Space


class Callback:
    """Observe or deliberately modify an optimization through lifecycle hooks.

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

    def on_iteration_begin(self, iteration: int, opt_model: opytimizer.Opytimizer) -> None:
        """Run before an update, using the cumulative one-based iteration counter.

        Args:
            iteration: Cumulative iteration counter, already incremented.
            opt_model: An instance of the optimization model.

        """

        pass

    def on_iteration_end(self, iteration: int, opt_model: opytimizer.Opytimizer) -> None:
        """Run after evaluation and history recording for a completed iteration.

        Args:
            iteration: Cumulative iteration counter, not the zero-based run index.
            opt_model: An instance of the optimization model.

        """

        pass

    def on_evaluate_before(self, *evaluate_args: Any) -> None:
        """Receive the evaluator's resolved arguments before it runs.

        Args:
            *evaluate_args: Current values resolved from the evaluator signature.

        """

        pass

    def on_evaluate_after(self, *evaluate_args: Any) -> None:
        """Receive the evaluator's resolved arguments after it returns normally.

        Args:
            *evaluate_args: Current values resolved from the evaluator signature.

        """

        pass

    def on_update_before(self, *update_args: Any) -> None:
        """Receive the updater's resolved arguments before it runs.

        Args:
            *update_args: Current values resolved from the updater signature.

        """

        pass

    def on_update_after(self, *update_args: Any) -> None:
        """Receive updater arguments after updating, before driver bound clipping.

        Args:
            *update_args: Current values resolved from the updater signature.

        """

        pass


class CheckpointCallback(Callback):
    """Save completed iterations using the model's dill serialization path.

    """

    def __init__(self, file_path: str | PathLike[str] | None = None, frequency: int = 0) -> None:
        """Configure a checkpoint filename and iteration interval.

        The iteration prefix is added to the filename without changing its directory.
        Parent directories must exist. A zero interval disables saving.

        Args:
            file_path: Text checkpoint path or path-like object.
            frequency: Interval in cumulative iterations.

        """

        if file_path is None:
            file_path = "checkpoint.pkl"
        if isinstance(file_path, PathLike):
            file_path = fspath(file_path)
        if not isinstance(file_path, str):
            raise TypeError("`file_path` must be a string or text path-like object.")

        if not isinstance(frequency, int):
            raise TypeError("`frequency` must be an integer.")
        if frequency < 0:
            raise ValueError("`frequency` must be non-negative.")

        self.file_path = file_path
        self.frequency = frequency

    def on_iteration_end(self, iteration: int, opt_model: opytimizer.Opytimizer) -> None:
        if self.frequency > 0 and iteration % self.frequency == 0:
            path = Path(self.file_path)
            opt_model.save(str(path.with_name(f"iter_{iteration}_{path.name}")))


class DiscreteSearchCallback(Callback):
    """Project each position component onto a variable's allowed values.

    """

    def __init__(self, allowed_values: list[list[int | float]] | None = None) -> None:
        """Configure one ordered set of discrete candidates per variable.

        Evaluation projects each dimension independently and ties select the first listed value.
        Task initialization validates nonempty candidate vectors and their bounds.

        Args:
            allowed_values: One nonempty list of candidate values per variable.

        """

        if allowed_values is None:
            allowed_values = []
        if not isinstance(allowed_values, list):
            raise TypeError("`allowed_values` must be a list.")

        self.allowed_values = allowed_values

    def on_task_begin(self, opt_model: opytimizer.Opytimizer) -> None:
        n_variables = opt_model.space.n_variables
        lower_bound = opt_model.space.lb
        upper_bound = opt_model.space.ub

        if len(self.allowed_values) != n_variables:
            raise ValueError(f"`allowed_values` must contain {n_variables} lists.")

        for values, lower, upper in zip(self.allowed_values, lower_bound, upper_bound):
            values = np.asarray(values)
            if values.ndim != 1 or values.size == 0:
                raise ValueError("`allowed_values` must contain non-empty vectors.")
            if not np.all((values >= lower) & (values <= upper)):
                raise ValueError("`allowed_values` must stay within the space bounds.")

    def on_evaluate_before(self, *evaluate_args: Any) -> None:
        space = evaluate_args[0]
        if not isinstance(space, Space):
            raise TypeError("`evaluate_args[0]` must be a Space.")

        for agent in space.agents:
            for i in range(agent.n_variables):
                values = np.asarray(self.allowed_values[i])
                min_value_idx = np.argmin(np.abs(agent.position[i, :, None] - values), axis=1)
                agent.position[i] = values[min_value_idx]

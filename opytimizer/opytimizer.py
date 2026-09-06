"""Optimization entry point."""

from __future__ import annotations

import operator
import time
from collections.abc import Callable, Sequence
from inspect import signature
from os import PathLike
from typing import Any, SupportsIndex

import dill

from opytimizer.core.optimizer import Optimizer
from opytimizer.core.space import Space
from opytimizer.utils.callback import Callback
from opytimizer.utils.history import History


def _emit(callbacks: Sequence[Callback] | None, event: str, *args: Any) -> None:
    if callbacks is None:
        return
    for callback in callbacks:
        getattr(callback, event)(*args)


class Opytimizer:
    """Coordinate a mutable population, optimization strategy, and objective.

    The supplied space and optimizer are retained, not copied. Construction
    compiles the optimizer once; repeated ``start`` calls continue its state and
    append history. Use separate instances for independent optimization tasks.
    """

    def __init__(
        self,
        space: Space,
        optimizer: Optimizer,
        function: Callable,
        save_agents: bool = False,
    ) -> None:
        """Bind an initialized space and compile the optimizer's state.

        Args:
            space: Initialized ``Space`` instance with its configured population.
            optimizer: Strategy instance. Compilation may reset its existing
                space-dependent buffers.
            function: Callable taking a position array of shape
                ``(n_variables, n_dimensions)`` and returning scalar fitness to
                minimize.
            save_agents: Whether to retain every agent's position and fitness at
                each completed iteration, in addition to the best agent.

        Raises:
            TypeError: If the supplied objects or history option have invalid types.
            RuntimeError: If the space has not initialized its population.
        """

        if not isinstance(space, Space):
            raise TypeError("`space` should be a Space")
        if len(space.agents) != space.n_agents:
            raise RuntimeError("`space` should be initialized")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("`optimizer` should be an Optimizer")
        if not callable(function):
            raise TypeError("`function` should be callable")

        history = History(save_agents=save_agents)

        self.space = space

        self.optimizer = optimizer
        self.optimizer.compile(space)

        self.function = function

        self.history = history

        self.iteration = 0
        self.total_iterations = 0

    @property
    def evaluate_args(self) -> list[Any]:
        """Current model attributes requested by the evaluator, in signature order."""

        args = signature(self.optimizer.evaluate).parameters

        return [getattr(self, v) for v in args]

    @property
    def update_args(self) -> list[Any]:
        """Current model attributes requested by the updater, in signature order."""

        args = signature(self.optimizer.update).parameters

        return [getattr(self, v) for v in args]

    def evaluate(self, callbacks: Sequence[Callback] | None = None) -> None:
        """Evaluate the population between the before/after evaluation hooks.

        Args:
            callbacks: Callbacks invoked in sequence order, or ``None``.

        Objective errors propagate and prevent the after hook from running.
        """

        _emit(callbacks, "on_evaluate_before", *self.evaluate_args)
        self.optimizer.evaluate(*self.evaluate_args)
        _emit(callbacks, "on_evaluate_after", *self.evaluate_args)

    def update(self, callbacks: Sequence[Callback] | None = None) -> None:
        """Update candidates, dispatch update hooks, then clip positions.

        Args:
            callbacks: Callbacks invoked in sequence order, or ``None``.

        ``on_update_after`` runs before the driver's bound clipping. Algorithm
        and callback errors propagate; state already changed is not rolled back.
        """

        _emit(callbacks, "on_update_before", *self.update_args)
        self.optimizer.update(*self.update_args)
        _emit(callbacks, "on_update_after", *self.update_args)

        self.space.clip_by_bound()

    def start(
        self,
        n_iterations: SupportsIndex = 1,
        callbacks: Sequence[Callback] | None = None,
    ) -> None:
        """Run additional iterations in place without recompiling the optimizer.

        Args:
            n_iterations: Non-negative integer budget. Python and NumPy integers
                are accepted. Zero evaluates the population and dispatches task
                hooks without performing an update.
            callbacks: Ordered callbacks for this invocation only. They are not
                registered for subsequent runs; supply them again after loading
                a checkpoint.

        Raises:
            TypeError: If the budget does not support integer indexing.
            ValueError: If the budget is negative.

        Notes:
            The loop sets ``iteration`` to a zero-based index for each update;
            before that it retains its previous value. ``total_iterations`` is
            incremented before each iteration-begin hook and is cumulative across
            calls. History is appended after evaluation, before iteration-end
            hooks. Elapsed time includes callbacks but excludes construction.

            Exceptions propagate without rollback. Task-end hooks and elapsed
            history are recorded only on normal completion. Results remain in
            ``space`` and ``history``; this method returns ``None``.
        """

        try:
            iterations = operator.index(n_iterations)
        except TypeError as error:
            raise TypeError("`n_iterations` should be an integer") from error
        if iterations < 0:
            raise ValueError("`n_iterations` should be >= 0")

        self.n_iterations = n_iterations
        callbacks = [] if callbacks is None else callbacks

        start = time.perf_counter()

        _emit(callbacks, "on_task_begin", self)

        self.evaluate(callbacks)

        for t in range(iterations):
            self.total_iterations += 1
            self.iteration = t

            _emit(callbacks, "on_iteration_begin", self.total_iterations, self)

            self.update(callbacks)
            self.evaluate(callbacks)

            self.history.dump(
                agents=self.space.agents, best_agent=self.space.best_agent
            )

            _emit(callbacks, "on_iteration_end", self.total_iterations, self)

        _emit(callbacks, "on_task_end", self)

        elapsed = time.perf_counter() - start
        self.history.dump(time=elapsed)

    def save(self, file_path: str | PathLike[str]) -> None:
        """Write this model's state to a dill checkpoint, replacing the file.

        Args:
            file_path: Text path or path-like object. Its parent directory must
                already exist.

        Filesystem and serialization errors propagate. The saved state includes
        the space, optimizer, objective, history, and counters. The driver does
        not register the callback sequence supplied to ``start`` as model state.
        """

        with open(file_path, "wb") as output_file:
            dill.dump(self, output_file)

    @classmethod
    def load(cls, file_path: str | PathLike[str]) -> Opytimizer:
        """Restore optimization state from a trusted dill checkpoint.

        Args:
            file_path: Text path or path-like object containing a saved model.

        Returns:
            The saved model, without recompiling its optimizer. Pass callbacks
            explicitly when calling ``start`` on the restored model.

        Warning:
            Dill uses pickle-based serialization and can execute code while
            loading. Never load checkpoints from untrusted sources.
        """

        with open(file_path, "rb") as input_file:
            return dill.load(input_file)

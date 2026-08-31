"""Optimization entry point."""

import time
from inspect import signature
from typing import Any, Callable, List, Optional

import dill

from opytimizer.core.optimizer import Optimizer
from opytimizer.core.space import Space
from opytimizer.utils.callback import Callback
from opytimizer.utils.history import History


def _emit(callbacks: Optional[List[Callback]], event: str, *args) -> None:
    for callback in callbacks or []:
        getattr(callback, event)(*args)


class Opytimizer:
    """An Opytimizer class holds all the information needed
    in order to perform an optimization task.

    """

    def __init__(
        self,
        space: Space,
        optimizer: Optimizer,
        function: Callable,
        save_agents: bool = False,
    ) -> None:
        """Initialization method.

        Args:
            space: Space-child instance.
            optimizer: Optimizer-child instance.
            function: Objective callable.
            save_agents: Saves all agents in the search space.

        """

        if not isinstance(space, Space):
            raise TypeError("`space` should be a Space")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("`optimizer` should be an Optimizer")
        if not callable(function):
            raise TypeError("`function` should be callable")

        self.space = space

        self.optimizer = optimizer
        self.optimizer.compile(space)

        self.function = function

        self.history = History(save_agents=save_agents)

        self.iteration = 0
        self.total_iterations = 0

    @property
    def evaluate_args(self) -> List[Any]:
        """Converts the optimizer `evaluate` arguments into real variables.

        Returns:
            (List[Any]): List of real-attribute variables.

        """

        args = signature(self.optimizer.evaluate).parameters

        return [getattr(self, v) for v in args]

    @property
    def update_args(self) -> List[Any]:
        """Converts the optimizer `update` arguments into real variables.

        Returns:
            (List[Any]): List of real-attribute variables.

        """

        args = signature(self.optimizer.update).parameters

        return [getattr(self, v) for v in args]

    def evaluate(self, callbacks: Optional[List[Callback]] = None) -> None:
        """Wraps the `evaluate` pipeline with its corresponding callbacks.

        Args:
            callbacks: List of callbacks.

        """

        _emit(callbacks, "on_evaluate_before", *self.evaluate_args)
        self.optimizer.evaluate(*self.evaluate_args)
        _emit(callbacks, "on_evaluate_after", *self.evaluate_args)

    def update(self, callbacks: Optional[List[Callback]] = None) -> None:
        """Wraps the `update` pipeline with its corresponding callbacks.

        Args:
            callbacks: List of callbacks.

        """

        _emit(callbacks, "on_update_before", *self.update_args)
        self.optimizer.update(*self.update_args)
        _emit(callbacks, "on_update_after", *self.update_args)

        # Regardless of callbacks or not, every update on the search space
        # must meet the bounds limits
        self.space.clip_by_bound()

    def start(
        self,
        n_iterations: int = 1,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        """Starts the optimization task.

        Args:
            n_iterations: Maximum number of iterations.
            callbacks: List of callbacks.

        """

        self.n_iterations = n_iterations
        callbacks = callbacks or []

        start = time.time()

        _emit(callbacks, "on_task_begin", self)

        self.evaluate(callbacks)

        for t in range(n_iterations):
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

        end = time.time()
        opt_time = end - start

        self.history.dump(time=opt_time)

    def save(self, file_path: str) -> None:
        """Saves the optimization model to a dill (pickle) file.

        Args:
            file_path: Path of file to be saved.

        """

        with open(file_path, "wb") as output_file:
            dill.dump(self, output_file)

    @classmethod
    def load(cls, file_path: str) -> "Opytimizer":
        """Loads the optimization model from a dill (pickle) file without needing
        to instantiate the class.

        Args:
            file_path: Path of file to be loaded.

        """

        with open(file_path, "rb") as input_file:
            opt_model = dill.load(input_file)

            return opt_model

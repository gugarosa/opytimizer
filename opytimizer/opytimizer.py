"""Optimization entry point.
"""

import time
from inspect import signature
from typing import Any, List, Optional

import dill
from tqdm import tqdm

import opytimizer.utils.exception as e
from opytimizer.core.function import Function
from opytimizer.core.optimizer import Optimizer
from opytimizer.core.space import Space
from opytimizer.utils import logging
from opytimizer.utils.callback import Callback, CallbackVessel
from opytimizer.utils.history import History

logger = logging.get_logger(__name__)


class Opytimizer:
    """An Opytimizer class holds all the information needed
    in order to perform an optimization task.

    """

    def __init__(
        self,
        space: Space,
        optimizer: Optimizer,
        function: Function,
        save_agents: bool = False,
    ) -> None:
        """Initialization method.

        Args:
            space: Space-child instance.
            optimizer: Optimizer-child instance.
            function: Function or Function-child instance.
            save_agents: Saves all agents in the search space.

        """

        logger.info("Creating class: Opytimizer.")

        self.space = space

        self.optimizer = optimizer
        self.optimizer.compile(space)

        self.function = function

        self.history = History(save_agents=save_agents)

        self.iteration = 0
        self.total_iterations = 0

        logger.debug(
            "Space: %s | Optimizer: %s| Function: %s.",
            self.space,
            self.optimizer,
            self.function,
        )
        logger.info("Class created.")

    @property
    def space(self) -> Space:
        """Space-child instance (SearchSpace, HyperComplexSpace, etc)."""

        return self._space

    @space.setter
    def space(self, space: Space) -> None:
        if not space.built:
            raise e.BuildError("`space` should be built before using Opytimizer")

        self._space = space

    @property
    def optimizer(self) -> Optimizer:
        """Optimizer-child instance (PSO, BA, etc)."""

        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        if not optimizer.built:
            raise e.BuildError("`optimizer` should be built before using Opytimizer")

        self._optimizer = optimizer

    @property
    def function(self) -> Function:
        """Function or Function-child instance (ConstrainedFunction, WeightedFunction, etc)."""

        return self._function

    @function.setter
    def function(self, function: Function) -> None:
        if not function.built:
            raise e.BuildError("`function` should be built before using Opytimizer")

        self._function = function

    @property
    def history(self) -> History:
        """Optimization history."""

        return self._history

    @history.setter
    def history(self, history: History) -> None:
        if not isinstance(history, History):
            raise e.TypeError("`history` should be a History")

        self._history = history

    @property
    def iteration(self) -> int:
        """Current iteration."""

        return self._iteration

    @iteration.setter
    def iteration(self, iteration: int) -> None:
        if not isinstance(iteration, int):
            raise e.TypeError("`iteration` should be an integer")
        if iteration < 0:
            raise e.ValueError("`iteration` should be >= 0")

        self._iteration = iteration

    @property
    def total_iterations(self) -> int:
        """Total number of iterations."""

        return self._total_iterations

    @total_iterations.setter
    def total_iterations(self, total_iterations: int) -> None:
        if not isinstance(total_iterations, int):
            raise e.TypeError("`total_iterations` should be an integer")
        if total_iterations < 0:
            raise e.ValueError("`total_iterations` should be >= 0")

        self._total_iterations = total_iterations

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

    def evaluate(self, callbacks: List[Callback]) -> None:
        """Wraps the `evaluate` pipeline with its corresponding callbacks.

        Args:
            callbacks: List of callbacks.

        """

        callbacks.on_evaluate_before(*self.evaluate_args)
        self.optimizer.evaluate(*self.evaluate_args)
        callbacks.on_evaluate_after(*self.evaluate_args)

    def update(self, callbacks: List[Callback]) -> None:
        """Wraps the `update` pipeline with its corresponding callbacks.

        Args:
            callback: List of callbacks.

        """

        callbacks.on_update_before(*self.update_args)
        self.optimizer.update(*self.update_args)
        callbacks.on_update_after(*self.update_args)

        # Regardless of callbacks or not, every update on the search space
        # must meet the bounds limits
        self.space.clip_by_bound()

    def start(
        self,
        n_iterations: int = 1,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        """Starts the optimization task.

        Args
            n_iterations: Maximum number of iterations.
            callback: List of callbacks.

        """

        logger.info("Starting optimization task.")

        self.n_iterations = n_iterations
        callbacks = CallbackVessel(callbacks)

        start = time.time()

        callbacks.on_task_begin(self)

        self.evaluate(callbacks)

        with tqdm(total=n_iterations, ascii=True) as b:
            for t in range(n_iterations):
                logger.to_file(f"Iteration {t+1}/{n_iterations}")

                self.total_iterations += 1
                self.iteration = t

                callbacks.on_iteration_begin(self.total_iterations, self)

                self.update(callbacks)
                self.evaluate(callbacks)

                b.set_postfix(fitness=self.space.best_agent.fit)
                b.update()

                self.history.dump(
                    agents=self.space.agents, best_agent=self.space.best_agent
                )

                callbacks.on_iteration_end(self.total_iterations, self)

                logger.to_file(f"Fitness: {self.space.best_agent.fit}")
                logger.to_file(f"Position: {self.space.best_agent.position}")

        callbacks.on_task_end(self)

        end = time.time()
        opt_time = end - start

        self.history.dump(time=opt_time)

        logger.info("Optimization task ended.")
        logger.info("It took %s seconds.", opt_time)

    def save(self, file_path: str) -> None:
        """Saves the optimization model to a dill (pickle) file.

        Args:
            file_path: Path of file to be saved.

        """

        with open(file_path, "wb") as output_file:
            dill.dump(self, output_file)

    @classmethod
    def load(cls, file_path: str) -> None:
        """Loads the optimization model from a dill (pickle) file without needing
        to instantiate the class.

        Args:
            file_path: Path of file to be loaded.

        """

        with open(file_path, "rb") as input_file:
            opt_model = dill.load(input_file)

            return opt_model

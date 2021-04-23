"""Optimization entry point.
"""

import time

from tqdm import tqdm

import opytimizer.utils.attribute as a
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.utils.callback import Callback
from opytimizer.utils.history import History

logger = l.get_logger(__name__)


class Opytimizer:
    """An Opytimizer class holds all the information needed
    in order to perform an optimization task.

    """

    def __init__(self, space, optimizer, function, callback=None,
                 store_best_only=True, iterations_per_snapshot=-1):
        """Initialization method.

        Args:
            space (Space): Space-child instance.
            optimizer (Optimizer): Optimizer-child instance.
            function (Function): Function or Function-child instance.
            callback (Callback): Pre-defined or customized callback object.
            store_best_only (bool): Stores only the history of best agent.
            iterations_per_snapshot (int): Saves a snapshot every `n` iterations.

        """

        logger.info('Creating class: Opytimizer.')

        # Space
        self.space = space

        # Optimizer
        self.optimizer = optimizer

        # Function
        self.function = function

        # Callback
        self.callback = callback or Callback()

        # History
        self.history = History(store_best_only, iterations_per_snapshot)

        # Total number of iterations
        self.total_iterations = 0

        # Logs the properties
        logger.debug('Space: %s | Optimizer: %s| Function: %s.',
                     self.space, self.optimizer, self.function)
        logger.info('Class created.')

    @property
    def space(self):
        """Space: Space-child instance (SearchSpace, HyperComplexSpace, etc).

        """

        return self._space

    @space.setter
    def space(self, space):
        if not space.built:
            raise e.BuildError(
                '`space` should be built before using Opytimizer')

        self._space = space

    @property
    def optimizer(self):
        """Optimizer: Optimizer-child instance (PSO, BA, etc).

        """

        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if not optimizer.built:
            raise e.BuildError(
                '`optimizer` should be built before using Opytimizer')

        self._optimizer = optimizer

    @property
    def function(self):
        """Function: Function or Function-child instance (ConstrainedFunction, WeightedFunction, etc).

        """

        return self._function

    @function.setter
    def function(self, function):
        if not function.built:
            raise e.BuildError(
                '`function` should be built before using Opytimizer')

        self._function = function

    @property
    def evaluate_args(self):
        """Converts the optimizer `evaluate` arguments into real variables.

        """

        return [a.rgetattr(self, v) for v in self.optimizer.args['evaluate']]

    @property
    def update_args(self):
        """Converts the optimizer `update` arguments into real variables.

        """

        return [a.rgetattr(self, v) for v in self.optimizer.args['update']]

    @property
    def history_kwargs(self):
        """Converts the optimizer `history` key-word arguments into real variables.

        """

        return {k: a.rgetattr(self, v) for k, v in self.optimizer.args['history'].items()}

    def evaluate(self):
        """Wraps the `evaluate` pipeline with its corresponding callbacks.

        """

        # Invokes the `on_evaluate_before` callback
        self.callback.on_evaluate_before(*self.evaluate_args)

        # Performs an evaluation over the search space
        self.optimizer.evaluate(*self.evaluate_args)

        # Invokes the `on_evaluate_after` callback
        self.callback.on_evaluate_after()

    def update(self):
        """Wraps the `update` pipeline with its corresponding callbacks.

        """

        # Invokes the `on_update_before` callback
        self.callback.on_update_before()

        # Performs an update over the search space
        self.optimizer.update(*self.update_args)

        # Invokes the `on_update_after` callback
        self.callback.on_update_after()

        # Regardless of callbacks or not, every update on the search space
        # must meet the bounds limits
        self.space.clip_by_bound()

    def start(self, n_iterations):
        """Starts the optimization task.

        Args
            n_iterations (int): Number of iterations.

        """

        logger.info('Starting optimization task.')

        # Additional properties
        self.n_iterations = n_iterations

        # Defines starting and ending points of iterations,
        # so when can handle multiple executions
        start_iteration = self.total_iterations
        end_iteration = self.total_iterations + n_iterations

        # Evaluates the search space
        self.evaluate()

        # Initializes a progress bar
        with tqdm(total=n_iterations) as b:
            # Loops through all iterations
            for t in range(start_iteration, end_iteration):
                logger.to_file(f'Iteration {t+1}/{end_iteration}')

                # Invokes the `on_iteration_begin` callback
                self.callback.on_iteration_begin(t+1, self.history)

                # Current iteration
                self.iteration = t

                # Updates the search space
                self.update()

                # Re-evaluates the search space
                self.evaluate()

                # Updates the progress bar status
                b.set_postfix(fitness=self.space.best_agent.fit)
                b.update()

                # Dumps key-word arguments in optimization history
                self.history.dump(**self.history_kwargs)

                # Invokes the `on_iteration_end` callback
                self.callback.on_iteration_end(t+1, self.history)

                logger.to_file(f'Fitness: {self.space.best_agent.fit}')
                logger.to_file(f'Position: {self.space.best_agent.position}')

        # Saves the number of total iterations that have be runned on object
        self.total_iterations += n_iterations

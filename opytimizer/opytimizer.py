"""Optimization entry point.
"""

import pickle
import time

from tqdm import tqdm

import opytimizer.utils.attribute as a
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.utils.callback import CallbackVessel
from opytimizer.utils.history import History

logger = l.get_logger(__name__)


class Opytimizer:
    """An Opytimizer class holds all the information needed
    in order to perform an optimization task.

    """

    def __init__(self, space, optimizer, function, store_only_best_agent=False):
        """Initialization method.

        Args:
            space (Space): Space-child instance.
            optimizer (Optimizer): Optimizer-child instance.
            function (Function): Function or Function-child instance.
            store_only_best_agent (bool): Stores only the best agent.

        """

        logger.info('Creating class: Opytimizer.')

        # Space
        self.space = space

        # Optimizer (and its additional variables)
        self.optimizer = optimizer
        self.optimizer.create_additional_vars(space)

        # Function
        self.function = function

        # Optimization history
        self.history = History(store_only_best_agent)

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
            raise e.BuildError('`space` should be built before using Opytimizer')

        self._space = space

    @property
    def optimizer(self):
        """Optimizer: Optimizer-child instance (PSO, BA, etc).

        """

        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if not optimizer.built:
            raise e.BuildError('`optimizer` should be built before using Opytimizer')

        self._optimizer = optimizer

    @property
    def function(self):
        """Function: Function or Function-child instance (ConstrainedFunction, WeightedFunction, etc).

        """

        return self._function

    @function.setter
    def function(self, function):
        if not function.built:
            raise e.BuildError('`function` should be built before using Opytimizer')

        self._function = function

    @property
    def history(self):
        """History: Optimization history.

        """

        return self._history

    @history.setter
    def history(self, history):
        if not isinstance(history, History):
            raise e.TypeError('`history` should be a History')

        self._history = history

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
        """Converts the optimizer `history` keyword arguments into real variables.

        """

        return {k: a.rgetattr(self, v) for k, v in self.optimizer.args['history'].items()}

    def evaluate(self, callbacks):
        """Wraps the `evaluate` pipeline with its corresponding callbacks.

        Args:
            callback (list): List of callbacks.

        """

        # Invokes the `on_evaluate_before` callback
        callbacks.on_evaluate_before(*self.evaluate_args)

        # Performs an evaluation over the search space
        self.optimizer.evaluate(*self.evaluate_args)

        # Invokes the `on_evaluate_after` callback
        callbacks.on_evaluate_after(*self.evaluate_args)

    def update(self, callbacks):
        """Wraps the `update` pipeline with its corresponding callbacks.

        Args:
            callback (list): List of callbacks.

        """

        # Invokes the `on_update_before` callback
        callbacks.on_update_before(*self.update_args)

        # Performs an update over the search space
        self.optimizer.update(*self.update_args)

        # Invokes the `on_update_after` callback
        callbacks.on_update_after(*self.update_args)

        # Regardless of callbacks or not, every update on the search space
        # must meet the bounds limits
        self.space.clip_by_bound()

    def start(self, n_iterations, callbacks=None):
        """Starts the optimization task.

        Args
            n_iterations (int): Number of iterations.
            callback (list): List of callbacks.

        """

        logger.info('Starting optimization task.')

        # Additional properties
        self.n_iterations = n_iterations
        callbacks = CallbackVessel(callbacks)

        # Triggers starting time
        start = time.time()

        # Evaluates the search space
        self.evaluate(callbacks)

        # Initializes a progress bar
        with tqdm(total=n_iterations) as b:
            # Loops through all iterations
            for t in range(n_iterations):
                logger.to_file(f'Iteration {t+1}/{n_iterations}')

                # Saves the number of total iterations and current iteration
                self.total_iterations += 1
                self.iteration = t

                # Invokes the `on_iteration_begin` callback
                callbacks.on_iteration_begin(self.total_iterations, self)

                # Updates the search space
                self.update(callbacks)

                # Re-evaluates the search space
                self.evaluate(callbacks)

                # Updates the progress bar status
                b.set_postfix(fitness=self.space.best_agent.fit)
                b.update()

                # Dumps keyword arguments to model's history
                self.history.dump(**self.history_kwargs)

                # Invokes the `on_iteration_end` callback
                callbacks.on_iteration_end(self.total_iterations, self)

                logger.to_file(f'Fitness: {self.space.best_agent.fit}')
                logger.to_file(f'Position: {self.space.best_agent.position}')

        # Stops the timer and calculates the optimization time
        end = time.time()
        opt_time = end - start

        # Dumps the elapsed time to model's history
        self.history.dump(time=opt_time)

        logger.info('Optimization task ended.')
        logger.info('It took %s seconds.', opt_time)

    def save(self, file_path):
        """Saves the optimization model to a pickle file.

        Args:
            file_path (str): Path of file to be saved.

        """

        # Opens an output file
        with open(file_path, 'wb') as output_file:
            # Dumps object to file
            pickle.dump(self, output_file)

    @classmethod
    def load(cls, file_path):
        """Loads the optimization model from a pickle file without needing
        to instantiate the class.

        Args:
            file_path (str): Path of file to be loaded.

        """

        # Opens an input file
        with open(file_path, "rb") as input_file:
            # Loads object from file
            opt_model = pickle.load(input_file)

            return opt_model

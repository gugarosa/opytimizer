"""Callbacks.
"""

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core.space import Space


class Callback:
    """A Callback class that handles additional variables and methods
    manipulation that are not provided by the library.

    """

    def __init__(self):
        """Initialization method."""

        pass

    def on_task_begin(self, opt_model):
        """Performs a callback whenever a task begins.

        Args:
            opt_model (Opytimizer): An instance of the optimization model.

        """

        pass

    def on_task_end(self, opt_model):
        """Performs a callback whenever a task ends.

        Args:
            opt_model (Opytimizer): An instance of the optimization model.

        """

        pass

    def on_iteration_begin(self, iteration, opt_model):
        """Performs a callback whenever an iteration begins.

        Args:
            iteration (int): Current iteration.
            opt_model (Opytimizer): An instance of the optimization model.

        """

        pass

    def on_iteration_end(self, iteration, opt_model):
        """Performs a callback whenever an iteration ends.

        Args:
            iteration (int): Current iteration.
            opt_model (Opytimizer): An instance of the optimization model.

        """

        pass

    def on_evaluate_before(self, *evaluate_args):
        """Performs a callback prior to the `evaluate` method."""

        pass

    def on_evaluate_after(self, *evaluate_args):
        """Performs a callback after the `evaluate` method."""

        pass

    def on_update_before(self, *update_args):
        """Performs a callback prior to the `update` method."""

        pass

    def on_update_after(self, *update_args):
        """Performs a callback after the `update` method."""

        pass


class CallbackVessel:
    """Wraps multiple callbacks in an ready-to-use class."""

    def __init__(self, callbacks):
        """Initialization method.

        Args:
            callbacks (list): List of Callback-based childs.

        """

        # Callbacks
        self.callbacks = callbacks or []

    @property
    def callbacks(self):
        """Space: List of Callback-based childs."""

        return self._callbacks

    @callbacks.setter
    def callbacks(self, callbacks):
        if not isinstance(callbacks, list):
            raise e.TypeError("`callbacks` should be a list")

        self._callbacks = callbacks

    def on_task_begin(self, opt_model):
        """Performs a list of callbacks whenever a task begins.

        Args:
            opt_model (Opytimizer): An instance of the optimization model.

        """

        for callback in self.callbacks:
            callback.on_task_begin(opt_model)

    def on_task_end(self, opt_model):
        """Performs a list of callbacks whenever a task ends.

        Args:
            opt_model (Opytimizer): An instance of the optimization model.

        """

        for callback in self.callbacks:
            callback.on_task_end(opt_model)

    def on_iteration_begin(self, iteration, opt_model):
        """Performs a list of callbacks whenever an iteration begins.

        Args:
            iteration (int): Current iteration.
            opt_model (Opytimizer): An instance of the optimization model.

        """

        for callback in self.callbacks:
            callback.on_iteration_begin(iteration, opt_model)

    def on_iteration_end(self, iteration, opt_model):
        """Performs a list of callbacks whenever an iteration ends.

        Args:
            iteration (int): Current iteration.
            opt_model (Opytimizer): An instance of the optimization model.

        """

        for callback in self.callbacks:
            callback.on_iteration_end(iteration, opt_model)

    def on_evaluate_before(self, *evaluate_args):
        """Performs a list of callbacks prior to the `evaluate` method."""

        for callback in self.callbacks:
            callback.on_evaluate_before(*evaluate_args)

    def on_evaluate_after(self, *evaluate_args):
        """Performs a list of callbacks after the `evaluate` method."""

        for callback in self.callbacks:
            callback.on_evaluate_after(*evaluate_args)

    def on_update_before(self, *update_args):
        """Performs a list of callbacks prior to the `update` method."""

        for callback in self.callbacks:
            callback.on_update_before(*update_args)

    def on_update_after(self, *update_args):
        """Performs a list of callbacks after the `update` method."""

        for callback in self.callbacks:
            callback.on_update_after(*update_args)


class CheckpointCallback(Callback):
    """A CheckpointCallback class that handles additional logging and
    model's checkpointing.

    """

    def __init__(self, file_path=None, frequency=0):
        """Initialization method.

        Args:
            file_path (str): Path of file to be saved.
            frequency (int): Interval between checkpoints.

        """

        super(CheckpointCallback, self).__init__()

        # File's path
        self.file_path = file_path or "checkpoint.pkl"

        # Interval between checkpoints
        self.frequency = frequency

    @property
    def file_path(self):
        """str: File's path."""

        return self._file_path

    @file_path.setter
    def file_path(self, file_path):
        if not isinstance(file_path, str):
            raise e.TypeError("`file_path` should be a string")

        self._file_path = file_path

    @property
    def frequency(self):
        """int: Interval between checkpoints."""

        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        if not isinstance(frequency, int):
            raise e.TypeError("`frequency` should be an integer")
        if frequency < 0:
            raise e.ValueError("`frequency` should be >= 0")

        self._frequency = frequency

    def on_iteration_end(self, iteration, opt_model):
        """Performs a callback whenever an iteration ends.

        Args:
            iteration (int): Current iteration.
            opt_model (Opytimizer): An instance of the optimization model.

        """

        # Checks if frequency is a positive number different than zero
        if self.frequency > 0:
            # If `mod` equals to zero
            # It means that current iteration must be checkpointed
            if iteration % self.frequency == 0:
                # Checkpoints the current model's state
                opt_model.save(f"iter_{iteration}_{self.file_path}")


class DiscreteSearchCallback(Callback):
    """A DiscreteSearchCallback class that handles mapping floating-point variables
    to discrete values.

    """

    def __init__(self, allowed_values=None):
        """Initialization method.

        Args:
            allowed_values (list): Possible values between lower and upper bounds that variables can be mapped.

        """

        super(DiscreteSearchCallback, self).__init__()

        # Allowed values between lower and upper bounds
        if allowed_values is not None:
            self.allowed_values = allowed_values
        else:
            self.allowed_values = []

    @property
    def allowed_values(self):
        """list: Allowed values between lower and upper bounds."""

        return self._allowed_values

    @allowed_values.setter
    def allowed_values(self, allowed_values):
        if not isinstance(allowed_values, list):
            raise e.TypeError("`allowed_values` should be a list")

        self._allowed_values = allowed_values

    def on_task_begin(self, opt_model):
        """Performs a callback whenever a task begins.

        Args:
            opt_model (Opytimizer): An instance of the optimization model.

        """

        # Gathers the number of variables, lower and upper bounds from search space
        n_variables = opt_model.space.n_variables
        lower_bound = opt_model.space.lb
        upper_bound = opt_model.space.ub

        assert (
            len(self.allowed_values) == n_variables
        ), f"`allowed_values` should have length equals to {n_variables}."
        assert np.all(
            [
                np.all((av >= lb) == (av <= ub))
                for av, lb, ub in zip(self.allowed_values, lower_bound, upper_bound)
            ]
        ), "Every value from `allowed_values` should be between `lower_bound` and `upper_bound`."

    def on_evaluate_before(self, *evaluate_args):
        """Performs a callback prior to the `evaluate` method."""

        space = evaluate_args[0]
        assert isinstance(
            space, Space
        ), "`evaluate_args[0]` is not derived from Space class."

        for agent in space.agents:
            for i in range(agent.n_variables):
                # Gathers the current closest allowed value and replaces agent's value
                min_value_idx = np.argmin(
                    abs(agent.position[i] - self.allowed_values[i])
                )
                agent.position[i] = self.allowed_values[i][min_value_idx]

"""Callbacks.
"""


class Callback:
    """A Callback class that handles additional variables and methods
    manipulation that are not provided by the library.

    """

    def __init__(self):
        """Initialization method.

        """

        pass

    def on_iteration_begin(self, iteration, opt_model):
        """Performs a callback whenever an iteration begins.

        """

        pass

    def on_iteration_end(self, iteration, opt_model):
        """Performs a callback whenever an iteration ends.

        """

        pass

    def on_evaluate_before(self, *evaluate_args):
        """Performs a callback prior to the `evaluate` method.

        """

        pass

    def on_evaluate_after(self, *evaluate_args):
        """Performs a callback after the `evaluate` method.

        """

        pass

    def on_update_before(self, *update_args):
        """Performs a callback prior to the `update` method.

        """

        pass

    def on_update_after(self, *update_args):
        """Performs a callback after the `update` method.

        """

        pass


class CallbackVessel:
    """Wraps multiple callbacks in an ready-to-use class.

    """

    def __init__(self, callbacks):
        """Initialization method.

        Args:
            callbacks (list): List of Callback-based childs.

        """

        # Callbacks
        self.callbacks = callbacks or []

    def on_iteration_begin(self, iteration, opt_model):
        """Performs a list of callbacks whenever an iteration begins.

        """

        # Iterates through all callbacks and invokes their method
        for callback in self.callbacks:
            callback.on_iteration_begin(iteration, opt_model)

    def on_iteration_end(self, iteration, opt_model):
        """Performs a list of callbacks whenever an iteration ends.

        """

        # Iterates through all callbacks and invokes their method
        for callback in self.callbacks:
            callback.on_iteration_end(iteration, opt_model)

    def on_evaluate_before(self, *evaluate_args):
        """Performs a list of callbacks prior to the `evaluate` method.

        """

        # Iterates through all callbacks and invokes their method
        for callback in self.callbacks:
            callback.on_evaluate_before(*evaluate_args)

    def on_evaluate_after(self, *evaluate_args):
        """Performs a list of callbacks after the `evaluate` method.

        """

        # Iterates through all callbacks and invokes their method
        for callback in self.callbacks:
            callback.on_evaluate_after(*evaluate_args)

    def on_update_before(self, *update_args):
        """Performs a list of callbacks prior to the `update` method.

        """

        # Iterates through all callbacks and invokes their method
        for callback in self.callbacks:
            callback.on_update_before(*update_args)

    def on_update_after(self, *update_args):
        """Performs a list of callbacks after the `update` method.

        """

        # Iterates through all callbacks and invokes their method
        for callback in self.callbacks:
            callback.on_update_after(*update_args)


class SnapshotCallback(Callback):
    """A SnaphotCallback class that handles addiitonal logging and
    model's snapshotting.

    """

    def __init__(self, iterations_per_snapshot=-1):
        """Initialization method.

        Args:
            iterations_per_snapshot (int): Saves a snapshot every `n` iterations.

        """

        # Overrides its parent class with the receiving arguments
        super(SnapshotCallback, self).__init__()

        # Iterations per snapshot
        self.iterations_per_snapshot = iterations_per_snapshot

    def on_iteration_end(self, iteration, opt_model):
        """Performs a callback whenever an iteration ends.

        """

        #
        if self.iterations_per_snapshot > 0:
            #
            save_snapshot = iteration % self.iterations_per_snapshot

            #
            if save_snapshot == 0:
                #
                opt_model.save(f'snapshot_iteration_{iteration}.pkl')

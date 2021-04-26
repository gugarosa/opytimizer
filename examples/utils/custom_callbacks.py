from opytimizer.utils.callback import Callback


class CustomCallback(Callback):
    """A CustomCallback can be created by override its parent `Callback` class
    and by implementing the desired logic in its available methods.

    """

    def __init__(self):
        """Initialization method for the customized callback.
        
        """

        # You only need to override its parent class
        super(CustomCallback).__init__()

    def on_iteration_begin(self, iteration, opt_model):
        """Called at the beginning of an iteration.

        """

        pass

    def on_iteration_end(self, iteration, opt_model):
        """Called at the end of an iteration.

        """

        pass

    def on_evaluate_before(self, *evaluate_args):
        """Called before the `evaluate` method.

        """

        pass

    def on_evaluate_after(self, *evaluate_args):
        """Called after the `evaluate` method.

        """

        pass

    def on_update_before(self, *update_args):
        """Called before the `update` method.

        """

        pass

    def on_update_after(self, *update_args):
        """Called after the `update` method.

        """

        pass

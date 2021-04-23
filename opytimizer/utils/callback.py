class Callback:
    """A Callback class that handles additional variables and methods
    manipulation that are not provided by the library.

    """

    def __init__(self):
        """
        """

    def on_iteration_begin(self, iteration, history):
        """Performs a callback whenever an iteration begins.

        """

        pass

    def on_iteration_end(self, iteration, history):
        """Performs a callback whenever an iteration ends.

        """

        #
        if history.iterations_per_snapshot > 0:
            #
            save_snapshot = iteration % history.iterations_per_snapshot

            #
            if save_snapshot == 0:
                #
                history.save(f'snapshot_iter_{iteration}.pkl')

    def on_evaluate_before(self, *args):
        """Performs a callback prior to the `evaluate` method.

        """

        pass

    def on_evaluate_after(self):
        """Performs a callback after the `evaluate` method.

        """

        pass

    def on_update_before(self):
        """Performs a callback prior to the `update` method.

        """

        pass

    def on_update_after(self):
        """Performs a callback after the `evaluate` method.

        """

        pass

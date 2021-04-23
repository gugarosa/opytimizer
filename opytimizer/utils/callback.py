class Callback:
    """A Callback class that handles additional variables and methods
    manipulation that are not provided by the library.

    """

    def on_iteration_begin(self, iteration, logs):
        """Performs a callback whenever an iteration begins.

        """

        # print(logs.best_agent)

        pass

    def on_iteration_end(self, iteration, logs):
        """Performs a callback whenever an iteration ends.

        """

        print(logs.best_agent)

        pass

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

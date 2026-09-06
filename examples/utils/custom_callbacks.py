# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Implement the callback hooks needed by an optimization task.

Task hooks receive the live driver. Iteration hooks also receive its iteration
number. Evaluation and update hooks receive the arguments dispatched to their
corresponding optimizer methods.

"""

from opytimizer.utils.callback import Callback


class CustomCallback(Callback):
    """Provide extension points for custom optimization lifecycle behavior.

    """

    def on_task_begin(self, opt_model):
        pass

    def on_task_end(self, opt_model):
        pass

    def on_iteration_begin(self, iteration, opt_model):
        pass

    def on_iteration_end(self, iteration, opt_model):
        pass

    def on_evaluate_before(self, *evaluate_args):
        pass

    def on_evaluate_after(self, *evaluate_args):
        pass

    def on_update_before(self, *update_args):
        pass

    def on_update_after(self, *update_args):
        pass

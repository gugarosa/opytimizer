"""Decorators.
"""

from functools import wraps

import opytimizer.math.hypercomplex as h


def hyper_spanning(lower_bound, upper_bound):
    """Spans a hyper-value between lower and upper bounds.

    Args:
        lower_bound (list | np.array): Lower bounds.
        upper_bound (list | np.array): Upper bounds.

    Returns:
        The output of the incoming objective function with a spanned input.

    """

    def _hyper_spanning(f_call):
        """Actually decorates the incoming objective function.

        Args:
            f_call (callable): Incoming objective function.

        Returns:
            The wrapped objective function.

        """

        @wraps(f_call)
        def __hyper_spanning(hyper_x):
            """Wraps the objective function for calculating its output.

            Args:
                hyper_x (np.array): Array of hyper-values.

            Returns:
                The objective function itself.

            """

            # Spans `hyper_x` between lower and upper bounds
            hyper_x = h.span(hyper_x, lower_bound, upper_bound)

            return f_call(hyper_x)

        return __hyper_spanning

    return _hyper_spanning


def pre_evaluation(f_call):
    """Pre-evaluates an objective function.

    Args:
        f_call (callable): Incoming objective function.

    Returns:
        The incoming objective function with its pre-evaluation.

    """

    @wraps(f_call)
    def _pre_evaluation(*args, **kwargs):
        """Wraps the objective function for calculating its pre-evaluation.

        Returns:
            The objective function itself.

        """

        # Check if there is a `hook` in keyword arguments
        if 'hook' in kwargs:
            # Applies it to a variable
            hook = kwargs['hook']

            # Check if variable is different than None
            if hook:
                # Calls the pre evaluation hook with the following arguments:
                # optimizer, space, function
                hook(args[0], args[1], args[2])

        return f_call(*args)

    return _pre_evaluation

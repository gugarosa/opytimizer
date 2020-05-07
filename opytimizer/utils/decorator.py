from functools import wraps

import opytimizer.math.hypercomplex as h


def hyper_spanning(lb, ub):
    """Spans a hyper-value between lower and upper bounds.

    Args:
        lb (list | np.array): Lower bounds.
        ub (list | np.array): Upper bounds.

    Returns:
        The output of the incoming objective function with a spanned input.

    """

    def _hyper_spanning(f):
        """Actually decorates the incoming objective function.

        Args:
            f (callable): Incoming objective function.

        Returns:
            The wrapped objective function.

        """

        @wraps(f)
        def __hyper_spanning(x):
            """Wraps the objective function for calculating its output.

            Args:
                x (np.array): Array of hyper-values.

            Returns:
                The objective function itself.

            """

            # Spans `x` between lower and upper bounds
            x = h.span(x, lb, ub)

            return f(x)

        return __hyper_spanning

    return _hyper_spanning


def pre_evaluation(f):
    """Pre-evaluates an objective function.

    Args:
        f (callable): Incoming objective function.

    Returns:
        The incoming objective function with its pre-evaluation.

    """

    @wraps(f)
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

        return f(*args)

    return _pre_evaluation

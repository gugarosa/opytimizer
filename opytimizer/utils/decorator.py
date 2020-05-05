from functools import wraps


def pre_evaluation(f):
    """Pre-evaluates an objective function.

    Args:
        f (callable): Incoming objective function.

    Returns:
        The incoming objective function with its pre-evaluation.

    """

    @wraps(f)
    def wrapper(*args, **kwargs):
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

    return wrapper

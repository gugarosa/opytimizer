"""Recursive-based attributes getter and setter.
"""

import functools


def rgetattr(obj, attr, *args):
    """Recursively gets an attribute.

    Reference:
        https://stackoverflow.com/a/31174427/12080653

    Args:
        obj (any): Object to be searched.
        attr (str): Attribute to be gathered.

    Returns:
        Recursive call to gather the attribute.

    """

    def _getattr(obj, attr):
        """Gets an attribute.

        Args:
            obj (any): Object to be searched.
            attr (str): Attribute to be gathered.

        Returns:
            Attribute.

        """

        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    """Recursively sets an attribute.

    Reference:
        https://stackoverflow.com/a/31174427/12080653

    Args:
        obj (any): Object to be modified.
        attr (str): Attribute to be created.
        value (any): Value to be set.

    Returns:
        Recursive call to set the attribute.

    """

    # Gathers the pre- and post-attributes
    pre, _, post = attr.rpartition('.')

    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

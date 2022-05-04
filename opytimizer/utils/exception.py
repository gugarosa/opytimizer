"""Exceptions.
"""

from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class Error(Exception):
    """A generic Error class derived from Exception.

    Essentially, it gets a class object and a message, and logs the error to the logger.

    """

    def __init__(self, cls: str, msg: str) -> None:
        """Initialization method.

        Args:
            cls: Class identifier.
            msg: Message to be logged.

        """

        super(Error, self).__init__()

        # Logs the error in a formatted way
        logger.error("%s: %s.", cls, msg)


class ArgumentError(Error):
    """An ArgumentError class for logging errors related to wrong number of provided arguments."""

    def __init__(self, error: str) -> None:
        """Initialization method.

        Args:
            error: Error message to be logged.

        """

        super(ArgumentError, self).__init__("ArgumentError", error)


class BuildError(Error):
    """A BuildError class for logging errors related to classes not being built."""

    def __init__(self, error: str) -> None:
        """Initialization method.

        Args:
            error: Error message to be logged.

        """

        super(BuildError, self).__init__("BuildError", error)


class SizeError(Error):
    """A SizeError class for logging errors related to wrong length or size of variables."""

    def __init__(self, error: str) -> None:
        """Initialization method.

        Args:
            error: Error message to be logged.

        """

        super(SizeError, self).__init__("SizeError", error)


class TypeError(Error):
    """A TypeError class for logging errors related to wrong type of variables."""

    def __init__(self, error: str) -> None:
        """Initialization method.

        Args:
            error: Error message to be logged.

        """

        super(TypeError, self).__init__("TypeError", error)


class ValueError(Error):
    """A ValueError class for logging errors related to wrong value of variables."""

    def __init__(self, error: str) -> None:
        """Initialization method.

        Args:
            error: Error message to be logged.

        """

        super(ValueError, self).__init__("ValueError", error)

"""Logging-based methods and helpers.
"""

import logging
import sys
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "opytimizer.log"
LOG_LEVEL = logging.DEBUG


class Logger(logging.Logger):
    """A customized Logger file that enables the possibility of only logging to file."""

    def to_file(self, msg: str, *args, **kwargs) -> None:
        """Logs the message only to the logging file.

        Args:
            msg: Message to be logged.

        """

        # Sets the console handler as critical level to disable console logging
        self.handlers[0].setLevel(logging.CRITICAL)

        # Logs the information
        self.info(msg, *args, **kwargs)

        # Re-enables the console handler logging
        self.handlers[0].setLevel(LOG_LEVEL)


def get_console_handler() -> StreamHandler:
    """Gets a console handler to handle logging into console.

    Returns:
        (StreamHandler): Handler to output information into console.

    """

    # Creates a stream handler for logger
    console_handler = StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)

    return console_handler


def get_timed_file_handler() -> TimedRotatingFileHandler:
    """Gets a timed file handler to handle logging into files.

    Returns:
        (TimedRotatingFileHandler): Handler to output information into timed files.

    """

    # Creates a timed rotating file handler for logger
    file_handler = TimedRotatingFileHandler(LOG_FILE, delay=True, when="midnight")
    file_handler.setFormatter(FORMATTER)

    return file_handler


def get_logger(logger_name: str) -> Logger:
    """Gets a logger and make it avaliable for further use.

    Args:
        logger_name: The name of the logger.

    Returns:
        (Logger): Logger instance.

    """

    # Defines a customized logger in order to have the possibility
    # of only logging to file when desired
    logging.setLoggerClass(Logger)

    # Creates a logger object
    logger = logging.getLogger(logger_name)

    # Sets an log level
    logger.setLevel(LOG_LEVEL)

    # Adds the desired handlers
    logger.addHandler(get_console_handler())
    logger.addHandler(get_timed_file_handler())

    # Do not propagate logs
    logger.propagate = False

    return logger

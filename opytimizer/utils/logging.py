"""Logging-based methods and helpers.
"""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOG_FILE = 'opytimizer.log'
LOG_LEVEL = logging.DEBUG


class Logger(logging.Logger):
    """Customized Logger class that enables the possibility
    of directly logging to files.

    """

    def to_file(self, msg, *args, **kwargs):
        """Logs the message directly to the logging file.

        Args:
            msg (str): Message to be logged.

        """

        # Sets the console handler as critical level to disable console logging
        self.handlers[0].setLevel(logging.CRITICAL)

        # Logs the information
        self.info(msg, *args, **kwargs)

        # Re-enables the console handler logging
        self.handlers[0].setLevel(LOG_LEVEL)


def get_console_handler():
    """Gets a console handler to handle console logging.

    Returns:
        Handler to output information into console.

    """

    # Creates a stream handler for logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)

    return console_handler


def get_timed_file_handler():
    """Gets a timed file handler to handle timed-files logging.

    Returns:
        Handler to output information into timed-files.

    """

    # Creates a timed rotating file handler for logger
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight', encoding='utf-8')
    file_handler.setFormatter(FORMATTER)

    return file_handler


def get_logger(logger_name):
    """Gets a log and makes it avaliable for further use.

    Args:
        logger_name (str): The name of the logger.

    Returns:
        Handler to output information into console.

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

    # Do not propagate any log
    logger.propagate = False

    return logger

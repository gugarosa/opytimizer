from opytimizer.utils import logging


def test_logging_to_file():
    logger = logging.get_logger(__name__)

    assert logger.to_file('msg') == None


def test_logging_get_console_handler():
    c = logging.get_console_handler()

    assert c != None


def test_logging_get_timed_file_handler():
    f = logging.get_timed_file_handler()

    assert f != None


def test_logging_get_logger():
    logger = logging.get_logger(__name__)

    assert logger.name == 'test_logging'

    assert logger.hasHandlers() == True

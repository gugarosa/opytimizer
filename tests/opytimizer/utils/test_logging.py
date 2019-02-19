import pytest

from opytimizer.utils import logging


def test_console_handler():
    c = logging.get_console_handler()

    assert c != None


def test_file_handler():
    f = logging.get_file_handler()

    assert f != None


def test_logger():
    logger = logging.get_logger(__name__)

    assert logger.name == 'test_logging'

    assert logger.hasHandlers() == True

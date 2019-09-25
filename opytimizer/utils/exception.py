import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Error(Exception):
    """
    """

    def __init__(self, cls, msg):
        """
        """

        #
        logger.error(f'{cls}: {msg}.')


class InvalidValueError(Error):
    """
    """

    def __init__(self, error):
        """
        """

        #
        super(InvalidValueError, self).__init__('InvalidValueError', error)

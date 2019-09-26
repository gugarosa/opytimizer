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


class ValueError(Error):
    """
    """

    def __init__(self, error):
        """
        """

        #
        super(ValueError, self).__init__('ValueError', error)

class TypeError(Error):
    """
    """

    def __init__(self, error):
        """
        """

        #
        super(TypeError, self).__init__('TypeError', error)

class ArgumentError(Error):
    """
    """

    def __init__(self, error):
        """
        """

        #
        super(ArgumentError, self).__init__('ArgumentError', error)

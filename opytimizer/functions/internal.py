import opytimizer.utils.logging as l
import py_expression_eval as math_parser

from opytimizer.core.function import Function

logger = l.get_logger(__name__)

class Internal(Function):

    def __init__(self):

        logger.info('Overriding Function with class: Internal')

        super(Internal, self).__init__(type='internal')

        logger.info('Internal created.')

    
    def build(self, function):
        """
        """

        logger.debug('Running method: build()')

        # Internal functions
        self.call = function

        # Set internal built variable to 'True'
        self._built = True

        logger.debug('Internal built with: ' + str(self.call))
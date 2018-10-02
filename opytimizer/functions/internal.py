import opytimizer.utils.logging as l
from opytimizer.core.function import Function

logger = l.get_logger(__name__)


class Internal(Function):
    """An Internal class, inherited from Function.
    It will server as the basis class for holding in-code related
    objective functions.

    Methods:
        build(function): Sets an external function point to a class
        attribute.

    """

    def __init__(self):
        """Initialization method.

        """

        logger.info('Overriding Function with class: Internal')

        # Overrides parent class with its own type
        super(Internal, self).__init__(type='internal')

        logger.info('Internal created.')

    def build(self, function):
        """This method will serve as the object building process.
        One can define several functions here that does not necessarily
        needs to be on its initialization.

        """

        logger.debug('Running method: build()')

        # Internal functions
        self.pointer = function

        # Set internal built variable to 'True'
        self._built = True

        logger.debug('Internal built with: ' + str(self.pointer))

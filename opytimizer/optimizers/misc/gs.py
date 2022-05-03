"""Grid-Search.
"""

from opytimizer.core import Optimizer
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class GS(Optimizer):
    """A GS class, inherited from Optimizer.

    This is the designed class to define grid search-related
    variables and methods.

    References:
        J. Bergstra and Y. Bengio. Random search for hyper-parameter optimization.
        Journal of machine learning research (2012).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> GS.")

        # Overrides its parent class with the receiving params
        super(GS, self).__init__()

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

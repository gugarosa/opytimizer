"""Non-Dominated Sorting.
"""

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core import Optimizer

logger = l.get_logger(__name__)


class NDS(Optimizer):
    """An NDS class, inherited from Optimizer.

    This is the designed class to define NDS-related
    variables and methods.

    References:
        To be added...

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Overrides its parent class with the receiving params
        super(NDS, self).__init__()

        self.eps = 1e-9

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

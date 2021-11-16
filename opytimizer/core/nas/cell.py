"""Cell.
"""

import time
import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Cell:
    """A Cell class for all graph-based optimization techniques.

    """

    def __init__(self, cell_type):
        """Initialization method.

        Args:
            operator (str): Operator to be applied when multiple inputs are provided.
            cell_type (str): Type of cell.
            params (dict): Key-value dictionary holding additional parameters.

        """

        self.type = cell_type
    
    def __call__(self):
        return 0.0
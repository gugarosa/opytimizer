"""Lightning Search Algorithm.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class LSA(Optimizer):
    """An LSA class, inherited from Optimizer.

    This is the designed class to define LSA-related
    variables and methods.

    References:
        H. Shareef, A. Ibrahim and A. Mutlag. Lightning search algorithm.
        Applied Soft Computing (2015).

    """

    def __init__(self, algorithm='LSA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> LSA.')

        # Override its parent class with the receiving hyperparams
        super(LSA, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

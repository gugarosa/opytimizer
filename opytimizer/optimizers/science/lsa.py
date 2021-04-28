"""Lightning Search Algorithm.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constant as c
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

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> LSA.')

        # Overrides its parent class with the receiving params
        super(LSA, self).__init__()

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

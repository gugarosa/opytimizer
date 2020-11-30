"""Artificial Flora.
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


class AF(Optimizer):
    """An AF class, inherited from Optimizer.

    This is the designed class to define AF-related
    variables and methods.

    References:
        L. Cheng, W. Xue-han and Y. Wang. Artificial flora (AF) optimization algorithm.
        Applied Sciences (2018).

    """

    def __init__(self, algorithm='AF', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> AF.')

        # Override its parent class with the receiving hyperparams
        super(AF, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

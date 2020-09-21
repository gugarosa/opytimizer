"""Artificial Butterfly Optimization.
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


class ABO(Optimizer):
    """An ABO class, inherited from Optimizer.

    This is the designed class to define ABO-related
    variables and methods.

    References:
        X. Qi, Y. Zhu and H. Zhang. A new meta-heuristic butterfly-inspired algorithm.
        Journal of Computational Science (2017).

    """

    def __init__(self, algorithm='ABO', hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> ABO.')

        # Override its parent class with the receiving hyperparams
        super(ABO, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

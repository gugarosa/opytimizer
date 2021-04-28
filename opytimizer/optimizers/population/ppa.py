"""Parasitism-Predation Algorithm.
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


class PPA(Optimizer):
    """A PPA class, inherited from Optimizer.

    This is the designed class to define PPA-related
    variables and methods.

    References:
        A. Mohamed et al. Parasitism – Predation algorithm (PPA): A novel approach for feature selection.
        Ain Shams Engineering Journal (2020).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> PPA.')

        # Overrides its parent class with the receiving params
        super(PPA, self).__init__()

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

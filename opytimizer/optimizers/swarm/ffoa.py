"""Fruit-Fly Optimization Algorithm.
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


class FFOA(Optimizer):
    """A FFOA class, inherited from Optimizer.

    This is the designed class to define FFOA-related
    variables and methods.

    References:
        W.-T. Pan. A new Fruit Fly Optimization Algorithm: Taking the financial distress model as an example.
        Knowledge-Based Systems (2012).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> FFOA.')

        # Overrides its parent class with the receiving params
        super(FFOA, self).__init__()

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

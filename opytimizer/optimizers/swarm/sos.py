"""Symbiotic Organisms Search.
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


class SOS(Optimizer):
    """An SOS class, inherited from Optimizer.

    This is the designed class to define SOS-related
    variables and methods.

    References:
        M.-Y. Cheng and D. Prayogo. Symbiotic Organisms Search: A new metaheuristic optimization algorithm.
        Computers & Structures (2014).

    """

    def __init__(self, algorithm='SOS', hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> SOS.')

        # Override its parent class with the receiving hyperparams
        super(SOS, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

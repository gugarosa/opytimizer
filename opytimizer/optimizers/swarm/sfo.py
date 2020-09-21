"""Sailfish Optimizer.
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


class SFO(Optimizer):
    """A SFO class, inherited from Optimizer.

    This is the designed class to define SFO-related
    variables and methods.

    References:
        S. Shadravan, H. Naji and V. Bardsiri.
        The Sailfish Optimizer: A novel nature-inspired metaheuristic algorithm
        for solving constrained engineering optimization problems.
        Engineering Applications of Artificial Intelligence (2019).
        
    """

    def __init__(self, algorithm='SFO', hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> SFO.')

        # Override its parent class with the receiving hyperparams
        super(SFO, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

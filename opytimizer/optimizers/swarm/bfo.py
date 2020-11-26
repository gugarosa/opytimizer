"""Bacterial Foraging Optimization.
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


class BFO(Optimizer):
    """A BFO class, inherited from Optimizer.

    This is the designed class to define BFO-related
    variables and methods.

    References:
        S. Das et al. Bacterial Foraging Optimization Algorithm: Theoretical Foundations, Analysis, and Applications.
        Foundations of Computational Intelligence (2009).

    """

    def __init__(self, algorithm='BFO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> BFO.')

        # Override its parent class with the receiving hyperparams
        super(BFO, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

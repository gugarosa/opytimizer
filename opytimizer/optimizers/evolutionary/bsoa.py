"""Backtracking Search Optimization Algorithm.
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


class BSOA(Optimizer):
    """A BSOA class, inherited from Optimizer.

    This is the designed class to define BSOA-related
    variables and methods.

    References:
        P. Civicioglu. Backtracking search optimization algorithm for numerical optimization problems.
        Applied Mathematics and Computation (2013).

    """

    def __init__(self, algorithm='BSOA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> BSOA.')

        # Override its parent class with the receiving hyperparams
        super(BSOA, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

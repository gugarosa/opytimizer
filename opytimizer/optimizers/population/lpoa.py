"""Lion Pride Optimization Algorithm.
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


class LPOA(Optimizer):
    """An LPOA class, inherited from Optimizer.

    This is the designed class to define LPOA-related
    variables and methods.

    References:
        A. Kaveh and S. Mahjoubi.
        Lion Pride Optimization Algorithm: A meta-heuristic method for global optimization problems.
        Scientia Iranica (2018).

    """

    def __init__(self, algorithm='LPOA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> LPOA.')

        # Override its parent class with the receiving hyperparams
        super(LPOA, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

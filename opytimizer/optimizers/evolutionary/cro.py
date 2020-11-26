"""Coral Reefs Optimization.
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


class CRO(Optimizer):
    """A CRO class, inherited from Optimizer.

    This is the designed class to define CRO-related
    variables and methods.

    References:
        S. Salcedo-Sanz et al.
        The Coral Reefs Optimization Algorithm: A Novel Metaheuristic for Efficiently Solving Optimization Problems.
        The Scientific World Journal (2014).

    """

    def __init__(self, algorithm='CRO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> CRO.')

        # Override its parent class with the receiving hyperparams
        super(CRO, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

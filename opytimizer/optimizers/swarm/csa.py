"""Crow Search Algorithm.
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


class CSA(Optimizer):
    """A CSA class, inherited from Optimizer.

    This is the designed class to define CSA-related
    variables and methods.

    References:
        A. Askarzadeh. A novel metaheuristic method for
        solving constrained engineering optimization problems: Crow search algorithm.
        Computers & Structures (2016).

    """

    def __init__(self, algorithm='CSA', hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> CSA.')

        # Override its parent class with the receiving hyperparams
        super(CSA, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

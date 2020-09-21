"""Queuing Search Algorithm.
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


class QSA(Optimizer):
    """A QSA class, inherited from Optimizer.

    This is the designed class to define QSA-related
    variables and methods.

    References:
        J. Zhang et al. Queuing search algorithm: A novel metaheuristic algorithm
        for solving engineering optimization problems.
        Applied Mathematical Modelling (2018).

    """

    def __init__(self, algorithm='QSA', hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> QSA.')

        # Override its parent class with the receiving hyperparams
        super(QSA, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

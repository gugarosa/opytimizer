"""Salp Swarm Algorithm.
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


class SSA(Optimizer):
    """A SSA class, inherited from Optimizer.

    This is the designed class to define SSA-related
    variables and methods.

    References:
        S. Mirjalili et al. Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems.
        Advances in Engineering Software (2017).

    """

    def __init__(self, algorithm='SSA', hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> SSA.')

        # Override its parent class with the receiving hyperparams
        super(SSA, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

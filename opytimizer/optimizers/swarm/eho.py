"""Elephant Herding Optimization.
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


class EHO(Optimizer):
    """An EHO class, inherited from Optimizer.

    This is the designed class to define EHO-related
    variables and methods.

    References:
        G.-G. Wang, S. Deb and L. Coelho. Elephant Herding Optimization.
        International Symposium on Computational and Business Intelligence (2015).

    """

    def __init__(self, algorithm='EHO', hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> EHO.')

        # Override its parent class with the receiving hyperparams
        super(EHO, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

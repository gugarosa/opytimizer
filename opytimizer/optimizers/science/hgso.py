"""Henry Gas Solubility Optimization.
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


class HGSO(Optimizer):
    """An HGSO class, inherited from Optimizer.

    This is the designed class to define HGSO-related
    variables and methods.

    References:
        F. Hashim et al. Henry gas solubility optimization: A novel physics-based algorithm.
        Future Generation Computer Systems (2019).

    """

    def __init__(self, algorithm='HGSO', hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> HGSO.')

        # Override its parent class with the receiving hyperparams
        super(HGSO, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

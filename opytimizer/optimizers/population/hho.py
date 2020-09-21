"""Harris Hawks Optimization.
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


class HHO(Optimizer):
    """An HHO class, inherited from Optimizer.

    This is the designed class to define HHO-related
    variables and methods.

    References:
        A. Heidari et al. Harris hawks optimization: Algorithm and applications.
        Future Generation Computer Systems (2019).

    """

    def __init__(self, algorithm='HHO', hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> HHO.')

        # Override its parent class with the receiving hyperparams
        super(HHO, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

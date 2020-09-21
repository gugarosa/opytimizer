"""Water Wave Optimization.
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


class WWO(Optimizer):
    """A WWO class, inherited from Optimizer.

    This is the designed class to define WWO-related
    variables and methods.

    References:
        Y.-J. Zheng. Water wave optimization: A new nature-inspired metaheuristic.
        Computers & Operations Research (2015).

    """

    def __init__(self, algorithm='WWO', hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> WWO.')

        # Override its parent class with the receiving hyperparams
        super(WWO, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

"""Brain Storm Optimization.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class BSO(Optimizer):
    """A BSO class, inherited from Optimizer.

    This is the designed class to define BSO-related
    variables and methods.

    References:
        Y. Shi. Brain Storm Optimization Algorithm.
        International Conference in Swarm Intelligence (2011).

    """

    def __init__(self, algorithm='BSO', params=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> BSO.')

        # Overrides its parent class with the receiving params
        super(BSO, self).__init__()

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

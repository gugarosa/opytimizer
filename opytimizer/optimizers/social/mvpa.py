"""Most Valuable Player Algorithm.
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


class MVPA(Optimizer):
    """A MVPA class, inherited from Optimizer.

    This is the designed class to define MVPA-related
    variables and methods.

    References:
        H. Bouchekara. Most Valuable Player Algorithm: a novel optimization algorithm inspired from sport.
        Operational Research (2017).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> MVPA.')

        # Overrides its parent class with the receiving params
        super(MVPA, self).__init__()

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

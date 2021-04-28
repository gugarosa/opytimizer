"""Sooty Tern Optimization Algorithm.
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


class STOA(Optimizer):
    """An STOA class, inherited from Optimizer.

    This is the designed class to define STOA-related
    variables and methods.

    References:
        G. Dhiman and A. Kaur. STOA: A bio-inspired based optimization algorithm for industrial engineering problems.
        Engineering Applications of Artificial Intelligence (2019).

    """

    def __init__(self, algorithm='STOA', params=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> STOA.')

        # Overrides its parent class with the receiving params
        super(STOA, self).__init__()

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

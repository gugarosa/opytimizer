"""Cohort Intelligence.
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


class CI(Optimizer):
    """A CI class, inherited from Optimizer.

    This is the designed class to define CI-related
    variables and methods.

    References:
        A. Kulkarni, G. Krishnasamy and A. Abraham. Cohort Intelligence: A Socio-inspired Optimization Method.
        Computational Intelligence and Complexity (2017).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> CI.')

        # Overrides its parent class with the receiving params
        super(CI, self).__init__()

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

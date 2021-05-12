"""Passing Vehicle Search.
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


class PVS(Optimizer):
    """A PVS class, inherited from Optimizer.

    This is the designed class to define PVS-related
    variables and methods.

    References:
        P. Savsani and V. Savsani. Passing vehicle search (PVS): A novel metaheuristic algorithm.
        Applied Mathematical Modelling (2016).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> PVS.')

        # Overrides its parent class with the receiving params
        super(PVS, self).__init__()

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

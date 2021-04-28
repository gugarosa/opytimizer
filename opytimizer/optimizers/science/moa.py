"""Magnetic Optimization Algorithm.
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


class MOA(Optimizer):
    """An MOA class, inherited from Optimizer.

    This is the designed class to define MOA-related
    variables and methods.

    References:
        M.-H. Tayarani and M.-R. Akbarzadeh. Magnetic-inspired optimization algorithms: Operators and structures.
        Swarm and Evolutionary Computation (2014).

    """

    def __init__(self, algorithm='MOA', params=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> MOA.')

        # Overrides its parent class with the receiving params
        super(MOA, self).__init__()

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

"""Interactive Search Algorithm.
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


class ISA(Optimizer):
    """An ISA class, inherited from Optimizer.

    This is the designed class to define ISA-related
    variables and methods.

    References:
        A. Mortazavi, V. Toğan and A. Nuhoğlu.
        Interactive search algorithm: A new hybrid metaheuristic optimization algorithm.
        Engineering Applications of Artificial Intelligence (2018).

    """

    def __init__(self, algorithm='ISA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> ISA.')

        # Override its parent class with the receiving hyperparams
        super(ISA, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

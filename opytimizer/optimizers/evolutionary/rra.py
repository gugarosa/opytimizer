"""Runner-Root Algorithm.
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


class RRA(Optimizer):
    """An RRA class, inherited from Optimizer.

    This is the designed class to define RRA-related
    variables and methods.

    References:
        F. Merrikh-Bayat.
        The runner-root algorithm: A metaheuristic for solving unimodal and
        multimodal optimization problems inspired by runners and roots of plants in nature.
        Applied Soft Computing (2015).

    """

    def __init__(self, algorithm='RRA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> RRA.')

        # Override its parent class with the receiving hyperparams
        super(RRA, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

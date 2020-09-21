"""Pigeon-Inspired Optimization.
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


class PIO(Optimizer):
    """A PIO class, inherited from Optimizer.

    This is the designed class to define PIO-related
    variables and methods.

    References:
        H. Duan and P. Qiao.
        Pigeon-inspired optimization:a new swarm intelligence optimizerfor air robot path planning.
        International Journal of IntelligentComputing and Cybernetics (2014).


    """

    def __init__(self, algorithm='PIO', hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> PIO.')

        # Override its parent class with the receiving hyperparams
        super(PIO, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

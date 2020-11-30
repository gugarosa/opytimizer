"""Flying Squirrel Optimizer.
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


class FSO(Optimizer):
    """A FSO class, inherited from Optimizer.

    This is the designed class to define FSO-related
    variables and methods.

    References:
        G. Azizyan et al.
        Flying Squirrel Optimizer (FSO): A novel SI-based optimization algorithm for engineering problems.
        Iranian Journal of Optimization (2019).

    """

    def __init__(self, algorithm='FSO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> FSO.')

        # Override its parent class with the receiving hyperparams
        super(FSO, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

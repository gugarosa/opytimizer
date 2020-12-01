"""Algorithm of the Innovative Gunner.
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


class AIG(Optimizer):
    """An AIG class, inherited from Optimizer.

    This is the designed class to define AIG-related
    variables and methods.

    References:
        P. Pijarski and P. Kacejko.
        A new metaheuristic optimization method: the algorithm of the innovative gunner (AIG).
        Engineering Optimization (2019).

    """

    def __init__(self, algorithm='AIG', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> AIG.')

        # Override its parent class with the receiving hyperparams
        super(AIG, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

"""Simplified Swarm Optimization.
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


class SSO(Optimizer):
    """A SSO class, inherited from Optimizer.

    This is the designed class to define SSO-related
    variables and methods.

    References:
        C. Bae et al. A new simplified swarm optimization (SSO) using exchange local search scheme.
        International Journal of Innovative Computing, Information and Control (2012).

    """

    def __init__(self, algorithm='SSO', hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> SSO.')

        # Override its parent class with the receiving hyperparams
        super(SSO, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

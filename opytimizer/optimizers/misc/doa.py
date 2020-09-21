"""Darcy Optimization Algorithm.
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


class DOA(Optimizer):
    """A DOA class, inherited from Optimizer.

    This is the designed class to define DOA-related
    variables and methods.

    References:
        F. Demir et al. A survival classification method for hepatocellular carcinoma patients
        with chaotic Darcy optimization method based feature selection.
        Medical Hypotheses (2020).

    """

    def __init__(self, algorithm='DOA', hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> DOA.')

        # Override its parent class with the receiving hyperparams
        super(DOA, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

import copy
import random

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core import agent
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class BWO(Optimizer):
    """A BWO class, inherited from Optimizer.

    This is the designed class to define BWO-related
    variables and methods.

    References:
        V. Hayyolalam and A. Kazem. 
        Black Widow Optimization Algorithm: A novel meta-heuristic approach for solving engineering optimization problems.
        Engineering Applications of Artificial Intelligence (2020). 

    """

    def __init__(self, algorithm='BWO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> BWO.')

        # Override its parent class with the receiving hyperparams
        super(BWO, self).__init__(algorithm=algorithm)

        # Procreating rate
        self.pp = 0.6

        # Cannibalism rate
        self.cr = 0.44

        # Mutation rate
        self.pm = 0.4

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def pp(self):
        """float: Procreating rate.

        """

        return self._pp

    @pp.setter
    def pp(self, pp):
        if not (isinstance(pp, float) or isinstance(pp, int)):
            raise e.TypeError('`pp` should be a float or integer')
        if pp < 0 or pp > 1:
            raise e.ValueError('`pp` should be between 0 and 1')

        self._pp = pp

    @property
    def cr(self):
        """float: Cannibalism rate.

        """

        return self._cr

    @cr.setter
    def cr(self, cr):
        if not (isinstance(cr, float) or isinstance(cr, int)):
            raise e.TypeError('`cr` should be a float or integer')
        if cr < 0:
            raise e.ValueError('`cr` should be >= 0')

        self._cr = cr

    @property
    def pm(self):
        """float: Mutation rate.

        """

        return self._pm

    @pm.setter
    def pm(self, pm):
        if not (isinstance(pm, float) or isinstance(pm, int)):
            raise e.TypeError('`pm` should be a float or integer')
        if pm < 0 or pm > 1:
            raise e.ValueError('`pm` should be between 0 and 1')

        self._pm = pm

    def _build(self, hyperparams):
        """This method serves as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.debug('Running private method: build().')

        # We need to save the hyperparams object for faster looking up
        self.hyperparams = hyperparams

        # If one can find any hyperparam inside its object,
        # set them as the ones that will be used
        if hyperparams:
            if 'pp' in hyperparams:
                self.pp = hyperparams['pp']
            if 'cr' in hyperparams:
                self.cr = hyperparams['cr']
            if 'pm' in hyperparams:
                self.pm = hyperparams['pm']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | '
            f'Hyperparameters: pp = {self.pp}, cr = {self.cr}, pm = {self.pm} | '
            f'Built: {self.built}.')

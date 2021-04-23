"""Opytimizer entry point.
"""

import time

from tqdm import tqdm

import opytimizer.utils.attribute as a
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Opytimizer:
    """An Opytimizer class holds all the information needed
    in order to perform an optimization task.

    """

    def __init__(self, space=None, optimizer=None, function=None):
        """Initialization method.

        Args:
            space (Space): A Space's object,
                where it has to be a child (e.g., SearchSpace, HyperComplexSpace, etc).
            optimizer (Optimizer): An Optimizer's object,
                where it has to be a child (e.g., PSO, BA, etc).
            function (Function): A Function's object,
                where it can be a child (e.g., WeightedFunction).

        """

        logger.info('Creating class: Opytimizer.')

        # Attaches the space to Opytimizer
        self.space = space

        # Attaches the optimizer
        self.optimizer = optimizer

        # Lastly, attaches the function
        self.function = function

        # We will log some important information
        logger.debug('Space: %s | Optimizer: %s| Function: %s.',
                     self.space, self.optimizer, self.function)

        logger.info('Class created.')

    @property
    def space(self):
        """Space: A Space's object, where it has to be a child (SearchSpace, HyperComplexSpace, etc).

        """

        return self._space

    @space.setter
    def space(self, space):
        if not space.built:
            raise e.BuildError('`space` should be built before using Opytimizer')

        self._space = space

    @property
    def optimizer(self):
        """Optimizer: An Optimizer's object, where it has to be a child (PSO, BA, etc).

        """

        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if not optimizer.built:
            raise e.BuildError('`optimizer` should be built before using Opytimizer')

        self._optimizer = optimizer

    @property
    def function(self):
        """Function: A Function's object, where it can be a child (WeightedFunction).

        """

        return self._function

    @function.setter
    def function(self, function):
        if not function.built:
            raise e.BuildError('`function` should be built before using Opytimizer')

        self._function = function

    def _get_optimizer_args(self):
        """
        """

        args = {}

        for k, v in self.optimizer.args.items():
            if type(v) == list:
                args[k] = []
                for item in v:
                    args[k].append(a.rgetattr(self, item))

        return args

            

    def start(self, n_iterations, store_best_only=False, pre_evaluate=None):
        """Starts the optimization task.

        Args
            store_best_only (bool): If True, only the best agent
                of each iteration is stored in History.
            pre_evaluate (callable): This function is executed
                before evaluating the function being optimized.

        Returns:
            A History object describing the agents position and best fitness values
                at each iteration throughout the optimization process.

        """

        logger.info('Starting optimization task.')

        #
        self.iteration = -1
        self.n_iterations = n_iterations

        #
        history = h.History(store_best_only)
        
        #
        args = self._get_optimizer_args()
        self.optimizer.evaluate(*args['evaluate'], hook=pre_evaluate)

        #
        with tqdm(total=n_iterations) as b:
            #
            for t in range(n_iterations):
                #
                self.iteration = t
                args = self._get_optimizer_args()

                logger.to_file(f'Iteration {t+1}/{n_iterations}')

                #
                self.optimizer._update(*args['update'])

                # Checking if agents meet the bounds limits
                self.space.clip_by_bound()

                # After the update, we need to re-evaluate the search space
                self.optimizer.evaluate(*args['evaluate'], hook=pre_evaluate)

                #
                # history.dump(*args['dump'])

                #
                b.set_postfix(fitness=self.space.best_agent.fit)
                b.update()

                logger.to_file(f'Fitness: {self.space.best_agent.fit}')
                logger.to_file(f'Position: {self.space.best_agent.position}')

        return history

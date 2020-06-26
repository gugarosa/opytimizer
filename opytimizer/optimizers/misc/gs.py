import numpy as np
from tqdm import tqdm

import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class GS(Optimizer):
    """A GS class, inherited from Optimizer.

    This is the designed class to define grid search-related
    variables and methods.

    References:
    J. Bergstra, and Y. Bengio. Random search for hyper-parameter optimization.
    Journal of machine learning research (2012).

    """

    def __init__(self, algorithm='GS', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> GS.')

        # Override its parent class with the receiving hyperparams
        super(GS, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build()

        logger.info('Class overrided.')

    def _build(self):
        """This method serves as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        """

        logger.debug('Running private method: build().')

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(f'Algorithm: {self.algorithm}  Built: {self.built}.')

    def run(self, space, function, store_best_only=False, pre_evaluation=None):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (bool): If True, only the best agent of each iteration is stored in History.
            pre_evaluation (callable): This function is executed before evaluating the function being optimized.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.file(f'Iteration {t+1}/{space.n_iterations}')

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, hook=pre_evaluation)

                # Every iteration, we need to dump agents and best agent
                history.dump(agents=space.agents, best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.file(f'Fitness: {space.best_agent.fit}')
                logger.file(f'Position: {space.best_agent.position}')

        return history

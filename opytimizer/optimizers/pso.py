import opytimizer.utils.common as c
import opytimizer.utils.logging as l
import opytimizer.utils.random as r
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class PSO(Optimizer):
    """A PSO class, inherited from Optimizer.

    This will be the designed class to define PSO-related
    variables and methods.

    Properties:
        w (float): Inertia weight parameter.

    Methods:
        _build(hyperparams): Sets an external function point to a class
        attribute.

    """

    def __init__(self, hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): An hyperparams dictionary containing key-value
            parameters to meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> PSO')

        # Override its parent class with the receiving hyperparams
        super(PSO, self).__init__(algorithm='PSO')

        # Default algorithm hyperparameters
        self.w = 2.0

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    def _build(self, hyperparams):
        """This method will serve as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            hyperparams (dict): An hyperparams dictionary containing key-value
            parameters to meta-heuristics.

        """

        logger.debug('Running private method: build()')

        # We need to save the hyperparams object for faster
        # looking up
        self.hyperparams = hyperparams

        # If one can find any hyperparam inside its object,
        # set them as the ones that will be used
        if self.hyperparams:
            if 'w' in self.hyperparams:
                self.w = self.hyperparams['w']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(f'Algorithm: {self.algorithm} | Hyperparameters: w = {self.w} | Built: {self.built}')

    def __update_position(self, agent, var):
        """
        """
        agent.position[var] = agent.position[var] * r.generate_uniform_random_number(0, 1)

    def _update(self, agents):
        """
        """
        for agent in agents:
            for var in range(agent.n_variables):
                self.__update_position(agent, var)
        pass


    def _evaluate(self, space, function):
        """
        """
        for agent in space.agents:
            fit = function.pointer(agent.position)
            if (fit < agent.fit):
                agent.fit = fit
            if (agent.fit < space.best_agent.fit):
                space.best_agent = agent

    def run(self, space, function):
        """
        """

        # Initial search space evaluation 
        self._evaluate(space, function)
        
        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1} out of {space.n_iterations}')

            # Updating agents' position
            self._update(space.agents)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            logger.info(f'Fitness: {space.best_agent.fit}')
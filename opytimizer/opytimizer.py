import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Opytimizer:
    """

    Properties:

    Methods:

    """

    def __init__(self, space, optimizer, function):
        """Initialization method.

        Args:
            space (Space):
            optimizer (Optimizer):
            function (Function):

        """

        logger.info('Initializing class: Opytimizer')

        # Checking if Space object is built
        if self._is_built(space):
            self.space = space

        # Checking if Optimizer object is built
        if self._is_built(optimizer):
            self.optimizer = optimizer

        # Checking if Function object is built
        if self._is_built(function):
            self.function = function

        # We will log some important information
        logger.info('Space: ' + str(self.space))
        logger.info('Optimizer: ' + str(self.optimizer))
        logger.info('Function: ' + str(self.function))
        logger.info('Opytimizer created.')

    def _is_built(self, obj):
        """
        """

        if obj._built:
            return True
        else:
            e = 'You are missing a built ' + obj.__class__.__name__ + ' object.'
            logger.error(e)
            raise RuntimeError(e)

    def run(self):
        """
        """

        self.optimizer.evaluate(self.space.agents, self.function)

        # for agent in self.space.agents:
        #     # Calculate fitness value over function
        #     fit = self.function.function(agent.position)

        #     # Check if fit is better than current agent's fit
        #     if fit < agent.fit:
        #         agent.fit = fit

        #     # Check if this newly agent is the best Space's agent
        #     self._is_best_agent(agent)

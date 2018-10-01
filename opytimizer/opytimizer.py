import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Opytimizer:
    """
    """

    def __init__(self, space=None, optimizer=None, function=None):
        """
        """

        logger.info('Initializing class: Opytimizer')

        # Initial variables declared as None
        self.space = None
        self.optimizer = None
        self.function = None

        # Space-related assignment
        if space._built:
            self.space = space
        else:
            e = 'You are missing a built Space object.'
            logger.error(e)
            raise RuntimeError(e)

        # Optimizer-related assignment
        if optimizer._built:
            self.optimizer = optimizer
        else:
            e = 'You are missing a built Optimizer object.'
            logger.error(e)
            raise RuntimeError(e)

        # Function-related assignment
        if function._built:
            self.function = function
        else:
            e = 'You are missing a built Function object.'
            logger.error(e)
            raise RuntimeError(e)

        # We will log some important information
        logger.info('Space: ' + str(self.space))
        logger.info('Optimizer: ' + str(self.optimizer))
        logger.info('Function: ' + str(self.function))
        logger.info('Opytimizer created.')

    def evaluate(self):
        """
        """

        for i in range(self.space.n_agents):
            print(self.function.function(self.space.agents[i].position))

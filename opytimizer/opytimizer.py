import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Opytimizer:
    """
    """

    def __init__(self, space, optimizer, function):
        """
        """

        logger.info('Initializing class: Opytimizer')

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

    def _is_best_agent(self, agent):
        """
        """
        
        if agent.fit < self.space.best_agent.fit:
            self.space.best_agent = agent

    def evaluate(self):
        """
        """

        for agent in self.space.agents:
            # Calculate fitness value over function
            fit = self.function.function(agent.position)
            
            # Check if fit is better than current agent's fit
            if fit < agent.fit:
                agent.fit = fit

            # Check if this newly agent is the best Space's agent
            self._is_best_agent(agent)

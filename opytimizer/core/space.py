import numpy as np
import opytimizer.utils.logging as l

from opytimizer.core.agent import Agent

logger = l.get_logger(__name__)


class Space:
    """
    """

    def __init__(self, n_agents=1):
        """
        """

        logger.info('Initializing Space ...')

        # External variables that can be accessed
        self.n_agents = n_agents
        self.agents = []
        self.best_agent = None

        # Internal use only
        self._built = False

        # We will log some important information
        logger.info('Space created.')

    def build(self, n_variables=2, n_dimensions=1):
        """
        """

        logger.debug('Running method: build()')

        for i in range(self.n_agents):
            self.agents.append(Agent(n_variables=n_variables, n_dimensions=n_dimensions))

        self._built = True

        logger.info('Space size: (' + str(self.n_agents) + ',' + str(n_variables) +
                    ',' + str(n_dimensions) + ')')

    def call(self):
        """
        """

        if not self._built:
            e = 'You need to call build() prior to call() method.'
            logger.error(e)
            raise RuntimeError(e)

        logger.debug('Running method: call()')


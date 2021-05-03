"""Boolean Manta Ray Foraging Optimization.
"""

import copy

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class BMRFO(Optimizer):
    """A BMRFO class, inherited from Optimizer.

    This is the designed class to define boolean MRFO-related
    variables and methods.

    References:
        Publication pending.

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> BMRFO.')

        # Overrides its parent class with the receiving params
        super(BMRFO, self).__init__()

        # Somersault foraging
        self.S = np.array([1])

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    @property
    def S(self):
        """float: Somersault foraging.

        """

        return self._S

    @S.setter
    def S(self, S):
        if not isinstance(S, np.ndarray):
            raise e.TypeError('`S` should be a numpy array')

        self._S = S

    def _cyclone_foraging(self, agents, best_position, i, iteration, n_iterations):
        """Performs the cyclone foraging procedure.

        Args:
            agents (list): List of agents.
            best_position (np.array): Global best position.
            i (int): Current agent's index.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        Returns:
            A new cyclone foraging.

        """

        # Generates binary random numbers
        r1 = r.generate_binary_random_number(best_position.shape)
        beta = r.generate_binary_random_number(best_position.shape)

        # Generates a uniform random number
        u = r.generate_uniform_random_number()

        # Checks if current iteration proportion is smaller than random generated number
        if iteration / n_iterations < u:
            # Generates binary random positions
            r_position = r.generate_binary_random_number(
                size=(agents[i].n_variables, agents[i].n_dimensions))

            # Checks if the index is equal to zero
            if i == 0:
                # Calculates the cyclone foraging
                partial_one = np.logical_or(r1, np.logical_xor(r_position, agents[i].position))
                partial_two = np.logical_or(beta, np.logical_xor(r_position, agents[i].position))
                cyclone_foraging = np.logical_and(r_position, np.logical_and(partial_one, partial_two))

            # If index is different than zero
            else:
                # Calculates the cyclone foraging
                partial_one = np.logical_or(r1, np.logical_xor(agents[i - 1].position, agents[i].position))
                partial_two = np.logical_or(beta, np.logical_xor(r_position, agents[i].position))
                cyclone_foraging = np.logical_and(r_position, np.logical_and(partial_one, partial_two))

        # If current iteration proportion is bigger than random generated number
        else:
            # Checks if the index is equal to zero
            if i == 0:
                # Calculates the cyclone foraging
                partial_one = np.logical_or(r1, np.logical_xor(best_position, agents[i].position))
                partial_two = np.logical_or(beta, np.logical_xor(best_position, agents[i].position))
                cyclone_foraging = np.logical_and(best_position, np.logical_and(partial_one, partial_two))

            # If index is different than zero
            else:
                # Calculates the cyclone foraging
                partial_one = np.logical_or(r1, np.logical_xor(agents[i - 1].position, agents[i].position))
                partial_two = np.logical_or(beta, np.logical_xor(best_position, agents[i].position))
                cyclone_foraging = np.logical_and(best_position, np.logical_and(partial_one, partial_two))

        return cyclone_foraging

    def _chain_foraging(self, agents, best_position, i):
        """Performs the chain foraging procedure.

        Args:
            agents (list): List of agents.
            best_position (np.array): Global best position.
            i (int): Current agent's index.

        Returns:
            A new chain foraging.

        """

        # Generates binary random numbers
        r1 = r.generate_binary_random_number(best_position.shape)
        alpha = r.generate_binary_random_number(best_position.shape)

        # Checks if the index is equal to zero
        if i == 0:
            # Calculates the chain foraging
            partial_one = np.logical_and(r1, np.logical_xor(best_position, agents[i].position))
            partial_two = np.logical_and(alpha, np.logical_xor(best_position, agents[i].position))
            chain_foraging = np.logical_or(agents[i].position, np.logical_or(partial_one, partial_two))

        # If index is different than zero
        else:
            # Calculates the chain foraging
            partial_one = np.logical_and(r1, np.logical_xor(agents[i - 1].position, agents[i].position))
            partial_two = np.logical_and(alpha, np.logical_xor(best_position, agents[i].position))
            chain_foraging = np.logical_or(agents[i].position, np.logical_or(partial_one, partial_two))

        return chain_foraging

    def _somersault_foraging(self, position, best_position):
        """Performs the somersault foraging procedure.

        Args:
            position (np.array): Agent's current position.
            best_position (np.array): Global best position.

        Returns:
            A new somersault foraging.

        """

        # Generates binary random numbers
        r1 = r.generate_binary_random_number(best_position.shape)
        r2 = r.generate_binary_random_number(best_position.shape)

        # Calculates the somersault foraging
        somersault_foraging = np.logical_or(position, np.logical_and(self.S, np.logical_xor(
            np.logical_xor(r1, best_position), np.logical_xor(r2, position))))

        return somersault_foraging

    def update(self, space, function, iteration, n_iterations):
        """Wraps chain, cyclone and somersault foraging updates over all agents and variables.

        Args:
            space (Space): Space containing agents and update-related information.
            function (Function): A Function object that will be used as the objective function.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Generates an uniform random number
            r1 = r.generate_uniform_random_number()

            # If random number is smaller than 1/2
            if r1 < 0.5:
                # Performs the cyclone foraging
                agent.position = self._cyclone_foraging(
                    space.agents, space.best_agent.position, i, iteration, n_iterations)

            # If random number is bigger than 1/2
            else:
                # Performs the chain foraging
                agent.position = self._chain_foraging(space.agents, space.best_agent.position, i)

            # Clips the agent's limits
            agent.clip_by_bound()

            # Evaluates the agent
            agent.fit = function(agent.position)

            # If new agent's fitness is better than best
            if agent.fit < space.best_agent.fit:
                # Replace the best agent's position and fitness with its copy
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)

        # Iterates through all agents
        for agent in space.agents:
            # Performs the somersault foraging
            agent.position = self._somersault_foraging(agent.position, space.best_agent.position)

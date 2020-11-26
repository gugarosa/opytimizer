"""Butterfly Optimization Algorithm.
"""

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class BOA(Optimizer):
    """A BOA class, inherited from Optimizer.

    This is the designed class to define BOA-related
    variables and methods.

    References:
        S. Arora and S. Singh. Butterfly optimization algorithm: a novel approach for global optimization.
        Soft Computing (2019).

    """

    def __init__(self, algorithm='BOA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> BOA.')

        # Override its parent class with the receiving hyperparams
        super(BOA, self).__init__(algorithm)

        # Sensor modality
        self.c = 0.01

        # Power exponent
        self.a = 0.1

        # Switch probability
        self.p = 0.8

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def c(self):
        """float: Sensor modality.

        """

        return self._c

    @c.setter
    def c(self, c):
        if not isinstance(c, (float, int)):
            raise e.TypeError('`c` should be a float or integer')
        if c < 0:
            raise e.ValueError('`c` should be >= 0')

        self._c = c

    @property
    def a(self):
        """float: Power exponent.

        """

        return self._a

    @a.setter
    def a(self, a):
        if not isinstance(a, (float, int)):
            raise e.TypeError('`a` should be a float or integer')
        if a < 0:
            raise e.ValueError('`a` should be >= 0')

        self._a = a

    @property
    def p(self):
        """float: Switch probability.

        """

        return self._p

    @p.setter
    def p(self, p):
        if not isinstance(p, (float, int)):
            raise e.TypeError('`p` should be a float or integer')
        if p < 0 or p > 1:
            raise e.ValueError('`p` should be between 0 and 1')

        self._p = p

    def _best_movement(self, agent_position, best_position, fragrance, random):
        """Updates the agent's position towards the best butterfly (eq. 2).

        Args:
            agent_position (np.array): Agent's current position.
            best_position (np.array): Best agent's current position.
            fragrance (np.array): Agent's current fragrance value.
            random (float): A random number between 0 and 1.

        Returns:
            A new position based on best movement.

        """

        # Calculates the new position based on best movement
        new_position = agent_position + (random ** 2 * best_position - agent_position) * fragrance

        return new_position

    def _local_movement(self, agent_position, j_position, k_position, fragrance, random):
        """Updates the agent's position using a local movement (eq. 3).

        Args:
            agent_position (np.array): Agent's current position.
            j_position (np.array): Agent `j` current position.
            k_position (np.array): Agent `k` current position.
            fragrance (np.array): Agent's current fragrance value.
            random (float): A random number between 0 and 1.

        Returns:
            A new position based on local movement.

        """

        # Calculates the new position based on local movement
        new_position = agent_position + (random ** 2 * j_position - k_position) * fragrance

        return new_position

    def _update(self, agents, best_agent, fragrance):
        """Method that wraps global and local pollination updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            fragrance (np.array): Array of fragrances.

        """

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Calculates fragrance for current agent (eq. 1)
            fragrance[i] = self.c * agent.fit ** self.a

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Generates a uniform random number
            r1 = r.generate_uniform_random_number()

            # If random number is smaller than switch probability
            if r1 < self.p:
                # Moves current agent towards the best one (eq. 2)
                agent.position = self._best_movement(
                    agent.position, best_agent.position, fragrance[i], r1)

            # If random number is bigger than switch probability
            else:
                # Generates a `j` index
                j = r.generate_integer_random_number(0, len(agents))

                #  Generates a `k` index
                k = r.generate_integer_random_number(0, len(agents), exclude_value=j)

                # Moves current agent using a local movement (eq. 3)
                agent.position = self._local_movement(
                    agent.position, agents[j].position, agents[k].position, fragrance[i], r1)

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

        # Instantiates an array of fragrances
        fragrance = np.zeros(space.n_agents)

        # Initial search space evaluation
        self._evaluate(space, function, hook=pre_evaluation)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.file(f'Iteration {t+1}/{space.n_iterations}')

                # Updating agents
                self._update(space.agents, space.best_agent, fragrance)

                # Checking if agents meet the bounds limits
                space.clip_limits()

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

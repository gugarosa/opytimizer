"""Jellyfish Search.
"""

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class JS(Optimizer):
    """A JS class, inherited from Optimizer.

    This is the designed class to define JS-related
    variables and methods.

    References:
        J.-S. Chou and D.-N. Truong. A novel metaheuristic optimizer inspired by behavior of jellyfish in ocean.
        Applied Mathematics and Computation (2020).

    """

    def __init__(self, algorithm='JS', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> JS.')

        # Override its parent class with the receiving hyperparams
        super(JS, self).__init__(algorithm)

        # Chaotic map coefficient
        self.eta = 4.0

        # Distribution coefficient
        self.beta = 3.0

        # Motion coefficient
        self.gamma = 0.1

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def eta(self):
        """float: Chaotic map coefficient.

        """

        return self._eta

    @eta.setter
    def eta(self, eta):
        if not isinstance(eta, (float, int)):
            raise e.TypeError('`eta` should be a float or integer')
        if eta < 0:
            raise e.ValueError('`eta` should be >= 0')

        self._eta = eta

    @property
    def beta(self):
        """float: Distribution coeffiecient.

        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        if not isinstance(beta, (float, int)):
            raise e.TypeError('`beta` should be a float or integer')
        if beta < 0:
            raise e.ValueError('`beta` should be >= 0')

        self._beta = beta

    @property
    def gamma(self):
        """float: Motion coeffiecient.

        """

        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        if not isinstance(gamma, (float, int)):
            raise e.TypeError('`gamma` should be a float or integer')
        if gamma < 0:
            raise e.ValueError('`gamma` should be >= 0')

        self._gamma = gamma

    def _initialize_chaotic_map(self, agents):
        """Initializes a set of agents using a logistic chaotic map.

        Args:
            agents (list): List of agents.

        """

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # If it is the first agent
            if i == 0:
                # Iterates through all decision variables
                for j in range(agent.n_variables):
                    # Calculates its position with a random uniform number
                    agent.position[j] = r.generate_uniform_random_number(size=agent.n_dimensions)

            # If it is not the first agent
            else:
                # Iterates through all decision variables
                for j in range(agent.n_variables):
                    # Calculates its position using logistic chaotic map (Eq. 18)
                    agent.position[j] = self.eta * agents[i - 1].position[j] * (1 - agents[i - 1].position[j])

    def _update(self, agents, best_agent, iteration, n_iterations):
        """Method that wraps the Jellyfish Search over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Iterates through all agents
        for agent in agents:
            # Generates an uniform random number
            r1 = r.generate_uniform_random_number()

            # Calculates the time control mechanism (Eq. 17)
            c = np.fabs((1 - iteration / n_iterations) * (2 * r1 - 1))

            # If time control mechanism is bigger or equal to 0.5
            if c >= 0.5:
                # Generates uniform random numbers
                r2 = r.generate_uniform_random_number()
                r3 = r.generate_uniform_random_number()

                # Calculates the mean location of all jellyfishes
                u = np.mean([agent.position for agent in agents])

                # Calculates the ocean current (Eq. 9)
                trend = best_agent.position - self.beta * r2 * u

                # Updates the location of current jellyfish (Eq. 11)
                agent.position += r3 * trend

            # If time control mechanism is smaller than 0.5
            else:
                # Generates a uniform random number
                r2 = r.generate_uniform_random_number()

                # If random number is bigger than 1 - time control mechanism
                if r2 > (1 - c):
                    # Generates uniform random number
                    r3 = r.generate_uniform_random_number()

                    # Update jellyfish's location with type A motion (Eq. 12)
                    agent.position += self.gamma * r3 * \
                        (np.expand_dims(agent.ub, -1) -
                         np.expand_dims(agent.lb, -1))

                # If random number is smaller
                else:
                    # Generates random numbers
                    j = r.generate_integer_random_number(0, len(agents))
                    r3 = r.generate_uniform_random_number()

                    # Checks if current fitness is bigger or equal to selected one
                    if agent.fit >= agents[j].fit:
                        # Determines its direction (Eq. 15 - top)
                        d = agents[j].position - agent.position

                    # If current fitness is smaller
                    else:
                        # Determines its direction (Eq. 15 - bottom)
                        d = agent.position - agents[j].position

                    # Update jellyfish's location with type B motion (Eq. 16)
                    agent.position += r3 * d

            # Clips the agent's limits
            agent.clip_limits()

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

        # Initializes current agents with a chaotic map
        self._initialize_chaotic_map(space.agents)

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
                self._update(space.agents, space.best_agent, t, space.n_iterations)

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

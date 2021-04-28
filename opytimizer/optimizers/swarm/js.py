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

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> JS.')

        # Overrides its parent class with the receiving params
        super(JS, self).__init__()

        # Chaotic map coefficient
        self.eta = 4.0

        # Distribution coefficient
        self.beta = 3.0

        # Motion coefficient
        self.gamma = 0.1

        # Builds the class
        self.build(params)

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
        if eta <= 0:
            raise e.ValueError('`eta` should be > 0')

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
        if beta <= 0:
            raise e.ValueError('`beta` should be > 0')

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
        if gamma <= 0:
            raise e.ValueError('`gamma` should be > 0')

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
                    # Calculates its position using logistic chaotic map (eq. 18)
                    agent.position[j] = self.eta * agents[i-1].position[j] * (1 - agents[i-1].position[j])

    def _ocean_current(self, agents, best_agent):
        """Calculates the ocean current (eq. 9).

        Args:
            agents (Agent): List of agents.
            best_agent (Agent): Best agent.

        Returns:
            A trend value for the ocean current.

        """

        # Generates an uniform random number
        r1 = r.generate_uniform_random_number()

        # Calculates the mean location of all jellyfishes
        u = np.mean([agent.position for agent in agents])

        # Calculates the ocean current (eq. 9)
        trend = best_agent.position - self.beta * r1 * u

        return trend

    def _motion_a(self, lb, ub):
        """Calculates type A motion (eq. 12).

        Args:
            lb (np.array): Array of lower bounds.
            ub (np.array): Array of upper bounds.

        Returns:
            A type A motion array.

        """

        # Generates an uniform random number
        r1 = r.generate_uniform_random_number()

        # Calculates type A motion
        motion = self.gamma * r1 * (np.expand_dims(ub, -1) - np.expand_dims(lb, -1))

        return motion

    def _motion_b(self, agent_i, agent_j):
        """Calculates type B motion (eq. 15).

        Args:
            agent_i (Agent): Current agent to be updated.
            agent_j (Agent): Selected agent.

        Returns:
            A type B motion array.

        """

        # Generates an uniform random number
        r1 = r.generate_uniform_random_number()

        # Checks if current fitness is bigger or equal to selected one
        if agent_i.fit >= agent_j.fit:
            # Determines its direction (eq. 15 - top)
            d = agent_j.position - agent_i.position

        # If current fitness is smaller
        else:
            # Determines its direction (eq. 15 - bottom)
            d = agent_i.position - agent_j.position

        # Calculates type B motion
        motion = r1 * d

        return motion

    def update(self, agents, best_agent, iteration, n_iterations):
        """Wraps the Jellyfish Search over all agents and variables.

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

            # Calculates the time control mechanism (eq. 17)
            c = np.fabs((1 - iteration / n_iterations) * (2 * r1 - 1))

            # If time control mechanism is bigger or equal to 0.5
            if c >= 0.5:
                # Calculates the ocean current (eq. 9)
                trend = self._ocean_current(agents, best_agent)

                # Generate a uniform random number
                r2 = r.generate_uniform_random_number()

                # Updates the location of current jellyfish (eq. 11)
                agent.position += r2 * trend

            # If time control mechanism is smaller than 0.5
            else:
                # Generates a uniform random number
                r2 = r.generate_uniform_random_number()

                # If random number is bigger than 1 - time control mechanism
                if r2 > (1 - c):
                    # Update jellyfish's location with type A motion (eq. 12)
                    agent.position += self._motion_a(agent.lb, agent.ub)

                # If random number is smaller
                else:
                    # Generates a random integer
                    j = r.generate_integer_random_number(0, len(agents))

                    # Updates jellyfish's location with type B motion (eq. 16)
                    agent.position += self._motion_b(agent, agents[j])

            # Clips the agent's limits
            agent.clip_by_bound()

    def run(self, space, function, store_best_only=False, pre_evaluate=None):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (bool): If True, only the best agent of each iteration is stored in History.
            pre_evaluate (callable): This function is executed before evaluating the function being optimized.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Initializes current agents with a chaotic map
        self._initialize_chaotic_map(space.agents)

        # Initial search space evaluation
        self._evaluate(space, function, hook=pre_evaluate)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.to_file(f'Iteration {t+1}/{space.n_iterations}')

                # Updates agents
                self._update(space.agents, space.best_agent, t, space.n_iterations)

                # Checking if agents meet the bounds limits
                space.clip_by_bound()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, hook=pre_evaluate)

                # Every iteration, we need to dump agents and best agent
                history.dump(agents=space.agents, best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.to_file(f'Fitness: {space.best_agent.fit}')
                logger.to_file(f'Position: {space.best_agent.position}')

        return history


class NBJS(JS):
    """An NBJS class, inherited from JS.

    This is the designed class to define NBJS-related
    variables and methods.

    References:
        Publication pending.

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: JS -> NBJS.')

        # Overrides its parent class with the receiving params
        super(NBJS, self).__init__(algorithm, params)

        logger.info('Class overrided.')

    def _motion_a(self, lb, ub):
        """Calculates type A motion.

        Args:
            lb (np.array): Array of lower bounds.
            ub (np.array): Array of upper bounds.

        Returns:
            A type A motion array.

        """

        # Generates an uniform random number
        r1 = r.generate_uniform_random_number()

        # Calculates type A motion
        motion = self.gamma * r1

        return motion

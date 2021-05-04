"""Sailfish Optimizer.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as ex
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class SFO(Optimizer):
    """A SFO class, inherited from Optimizer.

    This is the designed class to define SFO-related
    variables and methods.

    References:
        S. Shadravan, H. Naji and V. Bardsiri.
        The Sailfish Optimizer: A novel nature-inspired metaheuristic algorithm
        for solving constrained engineering optimization problems.
        Engineering Applications of Artificial Intelligence (2019).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> SFO.')

        # Overrides its parent class with the receiving params
        super(SFO, self).__init__()

        # Percentage of initial sailfishes
        self.PP = 0.1

        # Attack power coefficient
        self.A = 4

        # Attack power decrease
        self.e = 0.001

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    @property
    def PP(self):
        """float: Percentage of initial sailfishes.

        """

        return self._PP

    @PP.setter
    def PP(self, PP):
        if not isinstance(PP, (float, int)):
            raise ex.TypeError('`PP` should be a float or integer')
        if PP < 0 or PP > 1:
            raise ex.ValueError('`PP` should be between 0 and 1')

        self._PP = PP

    @property
    def A(self):
        """int: Attack power coefficient.

        """

        return self._A

    @A.setter
    def A(self, A):
        if not isinstance(A, int):
            raise ex.TypeError('`A` should be an integer')
        if A <= 0:
            raise ex.ValueError('`A` should be > 0')

        self._A = A

    @property
    def e(self):
        """float: Attack power decrease.

        """

        return self._e

    @e.setter
    def e(self, e):
        if not isinstance(e, (float, int)):
            raise ex.TypeError('`e` should be a float or integer')
        if e < 0:
            raise ex.ValueError('`e` should be >= 0')

        self._e = e

    def _generate_random_agent(self, agent):
        """Generates a new random-based agent.

        Args:
            agent (Agent): Agent to be copied.

        Returns:
            Random-based agent.

        """

        # Makes a deepcopy of agent
        a = copy.deepcopy(agent)

        # Fills agent with new random positions
        a.fill_with_uniform()

        return a

    def _calculate_lambda_i(self, n_sailfishes, n_sardines):
        """Calculates the lambda value (eq. 7).

        Args:
            n_sailfishes (int): Number of sailfishes.
            n_sardines (int): Number of sardines.

        Returns:
            Lambda value from current iteration.

        """

        # Calculates the prey density (eq. 8)
        PD = 1 - (n_sailfishes / (n_sailfishes + n_sardines))

        # Generates a random uniform number
        r1 = r.generate_uniform_random_number()

        # Calculates lambda
        lambda_i = 2 * r1 * PD - PD

        return lambda_i

    def _update_sailfish(self, agent, best_agent, best_sardine, lambda_i):
        """Updates the sailfish's position (eq. 6).

        Args:
            agent (Agent): Current agent's.
            best_agent (Agent): Best sailfish.
            best_sardine (Agent): Best sardine.
            lambda_i (float): Lambda value.

        Returns:
            An updated position.

        """

        # Generates a random uniform number
        r1 = r.generate_uniform_random_number()

        # Calculates the new position
        new_position = best_sardine.position - lambda_i * \
            (r1 * (best_agent.position - best_sardine.position) / 2 - agent.position)

        return new_position

    def update(self, agents, best_agent, function, sardines, iteration):
        """Wraps Sailfish Optimizer updates.

        Args:
            agents (list): List of agents (sailfishes).
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.
            sardines (list): List of agents (sardines).
            iteration (int): Current iteration.

        """

        # Gathers the best sardine
        best_sardine = sardines[0]

        # Calculates the number of sailfishes and sardines
        n_sailfishes = len(agents)
        n_sardines = len(sardines)

        # Calculates the number of decision variables
        n_variables = agents[0].n_variables

        # Iterates through every agent
        for agent in agents:
            # Calculates the lambda value
            lambda_i = self._calculate_lambda_i(n_sailfishes, n_sardines)

            # Updates agent's position
            agent.position = self._update_sailfish(agent, best_agent, best_sardine, lambda_i)

            # Clips agent's limits
            agent.clip_by_bound()

            # Re-evaluates agent's fitness
            agent.fit = function(agent.position)

        # Calculates the attack power (eq. 10)
        AP = np.fabs(self.A * (1 - 2 * iteration * self.e))

        # Checks if attack power is smaller than 0.5
        if AP < 0.5:
            # Calculates the number of sardines possible replacements (eq. 11)
            alpha = int(len(sardines) * AP)

            # Calculates the number of variables possible replacements (eq. 12)
            beta = int(n_variables * AP)

            # Generates a list of selected sardines
            selected_sardines = r.generate_integer_random_number(0, n_sardines, size=alpha)

            # Iterates through every selected sardine
            for i in selected_sardines:
                # Generates a list of selected variables
                selected_vars = r.generate_integer_random_number(0, n_variables, size=beta)

                # Iterates through every selected variable
                for j in selected_vars:
                    # Generates a uniform random number
                    r1 = r.generate_uniform_random_number()

                    # Updates the sardine's position (eq. 9)
                    sardines[i].position[j] = r1 * (best_agent.position[j] - sardines[i].position[j] + AP)

                # Clips sardine's limits
                sardines[i].clip_by_bound()

                # Re-calculates its fitness
                sardines[i].fit = function(sardines[i].position)

        # If attack power is bigger than 0.5
        else:
            # Iterates through every sardine
            for sardine in sardines:
                # Generates a uniform random number
                r1 = r.generate_uniform_random_number()

                # Updates the sardine's position (eq. 9)
                sardine.position = r1 * (best_agent.position - sardine.position + AP)

                # Clips sardine's limits
                sardine.clip_by_bound()

                # Re-calculates its fitness
                sardine.fit = function(sardine.position)

        # Sorts the population of agents (sailfishes) and sardines
        agents.sort(key=lambda x: x.fit)
        sardines.sort(key=lambda x: x.fit)

        # Iterates through every agent
        for agent in agents:
            # Iterates through every sardine
            for sardine in sardines:
                # If agent is worse than sardine (eq. 13)
                if agent.fit > sardine.fit:
                    # Copies sardine to agent
                    agent = copy.deepcopy(sardine)

                    break

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

        # Initializes a population of sardines
        sardines = [self._generate_random_agent(space.best_agent)
                    for _ in range(int(space.n_agents / self.PP))]

        # Iterates through every sardine
        for sardine in sardines:
            # Calculates its fitness
            sardine.fit = function(sardine.position)

        # Sorts the population of sardines
        sardines.sort(key=lambda x: x.fit)

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
                self._update(space.agents, space.best_agent, function, sardines, t)

                # Checks if agents meet the bounds limits
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

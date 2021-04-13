"""Coyote Optimization Algorithm.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class COA(Optimizer):
    """A COA class, inherited from Optimizer.

    This is the designed class to define COA-related
    variables and methods.

    References:
        J. Pierezan and L. Coelho. Coyote Optimization Algorithm: A New Metaheuristic for Global Optimization Problems.
        IEEE Congress on Evolutionary Computation (2018).

    """

    def __init__(self, algorithm='COA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> COA.')

        # Override its parent class with the receiving hyperparams
        super(COA, self).__init__(algorithm)

        # Number of packs
        self.n_p = 2

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def n_p(self):
        """int: Number of packs.

        """

        return self._n_p

    @n_p.setter
    def n_p(self, n_p):
        if not isinstance(n_p, int):
            raise e.TypeError('`n_p` should be an integer')
        if n_p <= 0:
            raise e.ValueError('`n_p` should be > 0')

        self._n_p = n_p

    def _get_agents_from_pack(self, agents, index, n_c):
        """Gets a set of agents from a specified pack.

        Args:
            agents (list): List of agents.
            index (int): Index of pack.
            n_c (int): Number of agents per pack.

        Returns:
            A sorted list of agents that belongs to the specified pack.

        """

        # Defines the starting and ending points
        start, end = index * n_c, (index + 1) * n_c

        return sorted(agents[start:end], key=lambda x: x.fit)

    def _transition_packs(self, agents, n_c):
        """Transits coyotes between packs (Eq. 4).

        Args:
            agents (list): List of agents.
            n_c (int): Number of coyotes per pack.

        """

        # Calculates the eviction probability
        p_e = 0.005 * len(agents)

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number()

        # If random number is smaller than eviction probability
        if r1 < p_e:
            # Gathers two random packs
            p1 = r.generate_integer_random_number(high=self.n_p)
            p2 = r.generate_integer_random_number(high=self.n_p)

            # Gathers two random coyotes
            c1 = r.generate_integer_random_number(high=n_c)
            c2 = r.generate_integer_random_number(high=n_c)

            # Calculates their indexes
            i = n_c * p1 + c1
            j = n_c * p2 + c2

            # Performs a swap betweeh them
            agents[i], agents[j] = copy.deepcopy(
                agents[j]), copy.deepcopy(agents[i])

    def _update(self, agents, function, n_c):
        """Method that wraps Coyote Optimization Algorithm over all agents and variables.

        Args:
            agents (list): List of agents.
            function (Function): A Function object that will be used as the objective function.
            n_c (int): Number of agents per pack.

        """

        # Iterates through all packs
        for i in range(self.n_p):
            # Gets the agents for the specified pack
            pack_agents = self._get_agents_from_pack(agents, i, n_c)

            # Gathers the alpha coyote (Eq. 5)
            alpha = pack_agents[0]

            # Computes the cultural tendency (Eq. 6)
            tendency = np.median(
                np.array([agent.position for agent in pack_agents]), axis=0)

            # Iterates through all coyotes in the pack
            for agent in pack_agents:
                # Makes a deepcopy of current coyote
                a = copy.deepcopy(agent)

                # Generates two random integers
                cr1 = r.generate_integer_random_number(high=len(pack_agents))
                cr2 = r.generate_integer_random_number(high=len(pack_agents))

                # Calculates the alpha and pack influences
                lambda_1 = alpha.position - pack_agents[cr1].position
                lambda_2 = tendency - pack_agents[cr2].position

                # Generates two random uniform numbers
                r1 = r.generate_uniform_random_number()
                r2 = r.generate_uniform_random_number()

                # Updates the social condition (Eq. 12)
                a.position += r1 * lambda_1 + r2 * lambda_2

                # Checks the agent's limits
                a.clip_limits()

                # Evaluates the agent (Eq. 13)
                a.fit = function(a.position)

                # If the new potision is better than current agent's position (Eq. 14)
                if a.fit < agent.fit:
                    # Replaces the current agent's position and fitness
                    agent.position = copy.deepcopy(a.position)
                    agent.fit = copy.deepcopy(a.fit)

            # Performs transition between packs (Eq. 4)
            self._transition_packs(agents, n_c)

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

        # Calculates the number of coyotes per pack
        n_c = space.n_agents // self.n_p

        # If number of coyotes per pack equals to zero
        if n_c == 0:
            # Throws an error
            raise e.ValueError(
                'Number of agents should be divisible by number of packs')

        # Initial search space evaluation
        self._evaluate(space, function, hook=pre_evaluate)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.to_file(f'Iteration {t+1}/{space.n_iterations}')

                # Updating agents
                self._update(space.agents, function, n_c)

                # Checking if agents meet the bounds limits
                space.clip_limits()

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

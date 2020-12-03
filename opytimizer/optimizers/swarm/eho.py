"""Elephant Herding Optimization.
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


class EHO(Optimizer):
    """An EHO class, inherited from Optimizer.

    This is the designed class to define EHO-related
    variables and methods.

    References:
        G.-G. Wang, S. Deb and L. Coelho. Elephant Herding Optimization.
        International Symposium on Computational and Business Intelligence (2015).

    """

    def __init__(self, algorithm='EHO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> EHO.')

        # Override its parent class with the receiving hyperparams
        super(EHO, self).__init__(algorithm)

        # Matriarch influence
        self.alpha = 0.5

        # Center influence
        self.beta = 0.1

        # Maximum number of clans
        self.n_clans = 10

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def alpha(self):
        """float: Matriarch influence.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, (float, int)):
            raise e.TypeError('`alpha` should be a float or integer')
        if alpha < 0 or alpha > 1:
            raise e.ValueError('`alpha` should be between 0 and 1')

        self._alpha = alpha

    @property
    def beta(self):
        """float: Center influence.

        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        if not isinstance(beta, (float, int)):
            raise e.TypeError('`beta` should be a float or integer')
        if beta < 0 or beta > 1:
            raise e.ValueError('`beta` should be between 0 and 1')

        self._beta = beta

    @property
    def n_clans(self):
        """int: Maximum number of clans.

        """

        return self._n_clans

    @n_clans.setter
    def n_clans(self, n_clans):
        if not isinstance(n_clans, int):
            raise e.TypeError('`n_clans` should be integer')
        if n_clans < 1:
            raise e.ValueError('`n_clans` should be > 0')

        self._n_clans = n_clans

    def _get_agents_from_clan(self, agents, index, n_ci):
        """Gets a set of agents from a specified clan.

        Args:
            agents (list): List of agents.
            index (int): Index of clan.
            n_ci (int): Number of agents per clan.

        Returns:
            A sorted list of agents that belongs to the specified clan.
        """

        # Defines the starting and ending points
        start, end = index * n_ci, (index + 1) * n_ci

        return sorted(agents[start:end], key=lambda x: x.fit)

    def _updating_operator(self, agents, centers, function, n_ci):
        """Performs the separating operator.

        Args:
            agents (list): List of agents.
            centers (list): List of centers.
            function (Function): A Function object that will be used as the objective function.
            n_ci (int): Number of agents per clan.

        """

        # Iterates through every clan
        for i in range(self.n_clans):
            # Gets the agents for the specified clan
            clan_agents = self._get_agents_from_clan(agents, i, n_ci)

            # Iterates through every agent in clan
            for j, agent in enumerate(clan_agents):
                # Creates a temporary agent
                a = copy.deepcopy(agent)

                # Generaters an uniform random number
                r1 = r.generate_uniform_random_number()

                # If it is the first agent in clan
                if j == 0:
                    # Updates its position (Eq. 2)
                    a.position = self.beta * centers[i]

                # If it is not the first (best) agent in clan
                else:
                    # Updates its position (Eq. 1)
                    a.position += self.alpha * (clan_agents[0].position - a.position) * r1

                # Checks the agent's limits
                a.clip_limits()

                # Evaluates the agent
                a.fit = function(a.position)

                # If the new potision is better than current agent's position
                if a.fit < agent.fit:
                    # Replaces the current agent's position
                    agent.position = copy.deepcopy(a.position)

                    # Also replaces its fitness
                    agent.fit = copy.deepcopy(a.fit)

    def _separating_operator(self, agents, n_ci):
        """Performs the separating operator.

        Args:
            agents (list): List of agents.
            n_ci (int): Number of agents per clan.

        """

        # Iterates through every clan
        for i in range(self.n_clans):
            # Gets the agents for the specified clan
            clan_agents = self._get_agents_from_clan(agents, i, n_ci)

            # Gathers the worst agent in clan
            worst = clan_agents[-1]

            # Generates a new position for the worst agent in clan (Eq. 4)
            for j, (lb, ub) in enumerate(zip(worst.lb, worst.ub)):
                # For each decision variable, we generate uniform random numbers
                worst.position[j] = r.generate_uniform_random_number(lb, ub, size=worst.n_dimensions)

    def _update(self, agents, function, n_ci):
        """Method that wraps Elephant Herd Optimization over all agents and variables.

        Args:
            agents (list): List of agents.
            function (Function): A Function object that will be used as the objective function.
            n_ci (int): Number of agents per clan.

        """

        # Instantiates a list of empty centers
        centers = []

        # Iterates through every clan
        for i in range(self.n_clans):
            # Gets the agents for the specified clan
            clan_agents = self._get_agents_from_clan(agents, i, n_ci)

            # Calculates the clan's center position
            clan_center = np.mean(np.array([agent.position for agent in clan_agents]), axis=0)

            # Appends the center position to the list
            centers.append(clan_center)

        # Performs the updating operator
        self._updating_operator(agents, centers, function, n_ci)

        # Performs the separating operators
        self._separating_operator(agents, n_ci)

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

        # Calculates the number of elephants per clan
        n_ci = space.n_agents // self.n_clans

        # If number of elephants per clan equals to zero
        if n_ci == 0:
            # Throws an error
            raise e.ValueError(
                'Number of agents should be divisible by number of clans')

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
                self._update(space.agents, function, n_ci)

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

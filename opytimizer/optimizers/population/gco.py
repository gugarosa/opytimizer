"""Germinal Center Optimization.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class GCO(Optimizer):
    """A GCO class, inherited from Optimizer.

    This is the designed class to define GCO-related
    variables and methods.

    References:
        C. Villaseñor et al. Germinal center optimization algorithm.
        International Journal of Computational Intelligence Systems (2018).

    """

    def __init__(self, algorithm='GCO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Override its parent class with the receiving hyperparams
        super(GCO, self).__init__(algorithm)

        # Cross-ratio
        self.CR = 0.7

        # Mutation factor
        self.F = 1.25

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def CR(self):
        """float: Cross-ratio parameter.

        """

        return self._CR

    @CR.setter
    def CR(self, CR):
        if not isinstance(CR, (float, int)):
            raise e.TypeError('`CR` should be a float or integer')
        if CR < 0 or CR > 1:
            raise e.ValueError('`CR` should be between 0 and 1')

        self._CR = CR

    @property
    def F(self):
        """float: Mutation factor.

        """

        return self._F

    @F.setter
    def F(self, F):
        if not isinstance(F, (float, int)):
            raise e.TypeError('`F` should be a float or integer')
        if F < 0:
            raise e.ValueError('`F` should be >= 0')

        self._F = F

    def _mutate_cell(self, agent, alpha, beta, gamma):
        """Mutates a new cell based on distinct cells (alg. 2).

        Args:
            agent (Agent): Current agent.
            alpha (Agent): 1st picked agent.
            beta (Agent): 2nd picked agent.
            gamma (Agent): 3rd picked agent.

        Returns:
            A mutated cell.

        """

        # Makes a deep copy of agent
        a = copy.deepcopy(agent)

        # For every decision variable
        for j in range(a.n_variables):
            # Generates a second random number
            r2 = r.generate_uniform_random_number()

            # If random number is smaller than cross-ratio
            if r2 < self.CR:
                # Updates the cell position
                a.position[j] = alpha.position[j] + self.F * (beta.position[j] - gamma.position[j])

        return a

    def _dark_zone(self, agents, function, life, counter):
        """Performs the dark-zone update process (alg. 1).

        Args:
            agents (list): List of agents.
            function (Function): A Function object that will be used as the objective function.
            life (np.array): An array holding each cell's current life.
            counter (np.array): An array holding each cell's copy counter.

        """

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # Generates the first random number, between 0 and 100
            r1 = r.generate_uniform_random_number(0, 100)

            # If random number is smaller than cell's life
            if r1 < life[i]:
                # Increases it counter by one
                counter[i] += 1

            # If it is not smaller
            else:
                # Resets the counter to one
                counter[i] = 1

            # Generates the counting distribution and pick three cells
            C = d.generate_choice_distribution(len(agents), counter / np.sum(counter), size=3)

            # Mutates a new cell based on current and pre-picked cells
            a = self._mutate_cell(agent, agents[C[0]], agents[C[1]], agents[C[2]])

            # Check agent limits
            a.clip_limits()

            # Calculates the fitness for the temporary position
            a.fit = function(a.position)

            # If new fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Copy its position to the agent
                agent.position = copy.deepcopy(a.position)

                # And also copy its fitness
                agent.fit = copy.deepcopy(a.fit)

                # Increases the life of cell by ten
                life[i] += 10

    def _light_zone(self, agents, life):
        """Performs the light-zone update process (alg. 1).

        Args:
            agents (list): List of agents.
            life (np.array): An array holding each cell's current life.

        """

        # Gathers a list of fitness from all agents
        fits = [agent.fit for agent in agents]

        # Calculates the minimum and maximum fitness
        min_fit, max_fit = np.min(fits), np.max(fits)

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # Resets the cell life to 10
            life[i] = 10

            # Calculates the current cell new life fitness
            life_fit = (agent.fit - max_fit) / (min_fit - max_fit + c.EPSILON)

            # Adds 10 * new life fitness to cell's life
            life[i] += 10 * life_fit

    def _update(self, agents, function, life, counter):
        """Method that wraps dark- and light-zone updates over all agents and variables.

        Args:
            agents (list): List of agents.
            function (Function): A Function object that will be used as the objective function.
            life (np.array): An array holding each cell's current life.
            counter (np.array): An array holding each cell's copy counter.

        """

        # Performs the dark-zone update process
        self._dark_zone(agents, function, life, counter)

        # Performs the light-zone update process
        self._light_zone(agents, life)

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

        # Instanciating array of lives
        life = r.generate_uniform_random_number(70, 70, space.n_agents)

        # Instanciating array of counters
        counter = np.ones(space.n_agents)

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
                self._update(space.agents, function, life, counter)

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

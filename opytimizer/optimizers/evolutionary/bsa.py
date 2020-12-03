"""Backtracking Search Optimization Algorithm.
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


class BSA(Optimizer):
    """A BSA class, inherited from Optimizer.

    This is the designed class to define BSOA-related
    variables and methods.

    References:
        P. Civicioglu. Backtracking search optimization algorithm for numerical optimization problems.
        Applied Mathematics and Computation (2013).

    """

    def __init__(self, algorithm='BSA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> BSA.')

        # Override its parent class with the receiving hyperparams
        super(BSA, self).__init__(algorithm)

        # Experience from previous generation
        self.F = 3.0

        # Number of non-crosses
        self.mix_rate = 1

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def F(self):
        """float: Experience from previous generation.

        """

        return self._F

    @F.setter
    def F(self, F):
        if not isinstance(F, (float, int)):
            raise e.TypeError('`F` should be a float or integer')

        self._F = F

    @property
    def mix_rate(self):
        """int: Number of non-crosses.

        """

        return self._mix_rate

    @mix_rate.setter
    def mix_rate(self, mix_rate):
        if not isinstance(mix_rate, int):
            raise e.TypeError('`mix_rate` should be an integer')
        if mix_rate < 0:
            raise e.ValueError('`mix_rate` should be > 0')

        self._mix_rate = mix_rate

    def _permute(self, agents, old_agents):
        """Performs the permuting operator.

        Args:
            agents (list): List of agents.
            old_agents (list): List of historical agents.

        """

        # Generates the `a` and `b` random uniform numbers
        a = r.generate_uniform_random_number()
        b = r.generate_uniform_random_number()

        # If `a` is smaller than `b`
        if a < b:
            # Performs a full copy on the historical population
            old_agents = copy.deepcopy(agents)

        # Generates two integers `i` and `j`
        i = r.generate_integer_random_number(high=len(agents))
        j = r.generate_integer_random_number(high=len(agents), exclude_value=i)

        # Swap the agents
        old_agents[i], old_agents[j] = copy.deepcopy(old_agents[j]), copy.deepcopy(old_agents[i])

    def _mutate(self, agents, old_agents):
        """Performs the mutation operator.

        Args:
            agents (list): List of agents.
            old_agents (list): List of historical agents.

        Returns:
            A list holding the trial agents.

        """

        # Makes a deepcopy to hold the trial agents
        trial_agents = copy.deepcopy(agents)

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number()

        # Iterates through all populations
        for (trial_agent, agent, old_agent) in zip(trial_agents, agents, old_agents):
            # Updates the new trial agent's position
            trial_agent.position = agent.position + self.F * r1 * (old_agent.position - agent.position)

            # Clips its limits
            trial_agent.clip_limits()

        return trial_agents

    def _crossover(self, agents, trial_agents):
        """Performs the crossover operator.

        Args:
            agents (list): List of agents.
            old_agents (list): List of trial agents.

        """

        # Defines the number of agents and variables
        n_agents = len(agents)
        n_variables = agents[0].n_variables

        # Creates a crossover map
        cross_map = np.ones((n_agents, n_variables))

        # Generates the `a` and `b` random uniform numbers
        a = r.generate_uniform_random_number()
        b = r.generate_uniform_random_number()

        # If `a` is smaller than `b`
        if a < b:
            # Iterates through all agents
            for i in range(n_agents):
                # Generates a uniform random number
                r1 = r.generate_uniform_random_number()

                # Calculates the number of non-crosses
                non_crosses = int(self.mix_rate * r1 * n_variables)

                # Iterates through the number of non-crosses
                for _ in range(non_crosses):
                    # Generates a random decision variable index
                    u = r.generate_integer_random_number(high=n_variables)

                    # Turn off the crossing on this specific point
                    cross_map[i][u] = 0

        # If `a` is bigger than `b`
        else:
            # Iterates through all agents
            for i in range(n_agents):
                # Generates a random decision variable index
                j = r.generate_integer_random_number(high=n_variables)

                # Turn off the crossing on this specific point
                cross_map[i][j] = 0

        # Iterates through all agents
        for i in range(n_agents):
            # Iterates through all decision variables
            for j in range(n_variables):
                # If it is supposed to cross according to the map
                if cross_map[i][j]:
                    # Makes a deepcopy on such position
                    trial_agents[i].position[j] = copy.deepcopy(agents[i].position[j])

    def _update(self, agents, function, old_agents):
        """Method that wraps the update pipeline over all agents and variables.

        Args:
            agents (list): List of agents.
            function (Function): A function object.
            old_agents (list): List of historical agents.

        """

        # Performs the permuting operator
        self._permute(agents, old_agents)

        # Calculate the trial agents based on the mutation operator
        trial_agents = self._mutate(agents, old_agents)

        # Performs the crossover
        self._crossover(agents, trial_agents)

        # Iterates through all agents and trial agents
        for (agent, trial_agent) in zip(agents, trial_agents):
            # Calculates the trial agent's fitness
            trial_agent.fit = function(trial_agent.position)

            # If its fitness is better than agent's fitness
            if trial_agent.fit < agent.fit:
                # Copies the trial agent's position to the agent's
                agent.position = copy.deepcopy(trial_agent.position)

                # Also copies its fitness
                agent.fit = copy.deepcopy(trial_agent.fit)

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

        # Makes a deepcopy of agents into the historical population
        old_agents = copy.deepcopy(space.agents)

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
                self._update(space.agents, function, old_agents)

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

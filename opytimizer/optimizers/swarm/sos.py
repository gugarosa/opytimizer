"""Symbiotic Organisms Search.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class SOS(Optimizer):
    """An SOS class, inherited from Optimizer.

    This is the designed class to define SOS-related
    variables and methods.

    References:
        M.-Y. Cheng and D. Prayogo. Symbiotic Organisms Search: A new metaheuristic optimization algorithm.
        Computers & Structures (2014).

    """

    def __init__(self, algorithm='SOS', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> SOS.')

        # Override its parent class with the receiving hyperparams
        super(SOS, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    def _mutualism(self, agent_i, agent_j, best_agent, function):
        """Performs the mutualism operation.

        Args:
            agent_i (Agent): Selected `i` agent.
            agent_j (Agent): Selected `j` agent.
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.

        """

        # Copies temporary agents from `i` and `j`
        a = copy.deepcopy(agent_i)
        b = copy.deepcopy(agent_j)

        # Calculates the mutual vector (Eq. 3)
        mutual_vector = (agent_i.position + agent_j.position) / 2

        # Calculates the benefitial factors
        BF_1, BF_2 = np.random.choice([1, 2], 2, replace=False)

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number()

        # Re-calculates the new positions (Eq. 1 and 2)
        a.position += r1 * (best_agent.position - mutual_vector * BF_1)
        b.position += r1 * (best_agent.position - mutual_vector * BF_2)

        # Checks their limits
        a.clip_limits()
        b.clip_limits()

        # Evaluates both agents
        a.fit = function(a.position)
        b.fit = function(b.position)

        # If new position is better than agent's `i` position
        if a.fit < agent_i.fit:
            # Replaces the agent's `i` position
            agent_i.position = copy.deepcopy(a.position)

            # Also replaces its fitness
            agent_i.fit = copy.deepcopy(a.fit)

        # If new position is better than agent's `j` position
        if b.fit < agent_j.fit:
            # Replaces the agent's `j` position
            agent_j.position = copy.deepcopy(b.position)

            # Also replaces its fitness
            agent_j.fit = copy.deepcopy(b.fit)

    def _commensalism(self, agent_i, agent_j, best_agent, function):
        """Performs the commensalism operation.

        Args:
            agent_i (Agent): Selected `i` agent.
            agent_j (Agent): Selected `j` agent.
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.

        """

        # Copies a temporary agent from `i`
        a = copy.deepcopy(agent_i)

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number(-1, 1)

        # Updates the agent's position (Eq. 4)
        a.position += r1 * (best_agent.position - agent_j.position)

        # Checks its limits
        a.clip_limits()

        # Evaluates its new position
        a.fit = function(a.position)

        # If the new position is better than the current agent's position
        if a.fit < agent_i.fit:
            # Replaces the current agent's position
            agent_i.position = copy.deepcopy(a.position)

            # Also replaces its fitness
            agent_i.fit = copy.deepcopy(a.fit)

    def _parasitism(self, agent_i, agent_j, function):
        """Performs the parasitism operation.

        Args:
            agent_i (Agent): Selected `i` agent.
            agent_j (Agent): Selected `j` agent.
            function (Function): A Function object that will be used as the objective function.

        """

        # Creates a temporary parasite agent
        p = copy.deepcopy(agent_i)

        # Generates a integer random number
        r1 = r.generate_integer_random_number(0, agent_i.n_variables)

        # Updates its position on selected variable with a uniform random number
        p.position[r1] = r.generate_uniform_random_number(p.lb[r1], p.ub[r1])

        # Checks its limits
        p.clip_limits()

        # Evaluates its position
        p.fit = function(p.position)

        # If the new potision is better than agent's `j` position
        if p.fit < agent_j.fit:
            # Replaces the agent's `j` position
            agent_j.position = copy.deepcopy(p.position)

            # Also replaces its fitness
            agent_j.fit = copy.deepcopy(p.fit)

    def _update(self, agents, best_agent, function):
        """Method that wraps Symbiotic Organisms Search. over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.

        """

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Generates a random integer for mutualism and performs it
            j = r.generate_integer_random_number(0, len(agents), exclude_value=i)
            self._mutualism(agent, agents[j], best_agent, function)

            # Re-generates a random integer for commensalism and performs it
            j = r.generate_integer_random_number(0, len(agents), exclude_value=i)
            self._commensalism(agent, agents[j], best_agent, function)

            # Re-generates a random integer for parasitism and performs it
            j = r.generate_integer_random_number(0, len(agents), exclude_value=i)
            self._parasitism(agent, agents[j], function)

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
                self._update(space.agents, space.best_agent, function)

                # Checking if agents meets the bounds limits
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

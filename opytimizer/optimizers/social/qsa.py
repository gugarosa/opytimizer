"""Queuing Search Algorithm.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class QSA(Optimizer):
    """A QSA class, inherited from Optimizer.

    This is the designed class to define QSA-related
    variables and methods.

    References:
        J. Zhang et al. Queuing search algorithm: A novel metaheuristic algorithm
        for solving engineering optimization problems.
        Applied Mathematical Modelling (2018).

    """

    def __init__(self, algorithm='QSA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> QSA.')

        # Override its parent class with the receiving hyperparams
        super(QSA, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    def _calculate_queue(self, n_agents, t1, t2, t3):
        """
        """

        # Checks if potential service time is bigger than `1e-6`
        if t1 > 1e-6:
            # Calculates the proportion of agents in first, second and third queues
            n1 = (1 / t1) / ((1 / t1) + (1 / t2) + (1 / t3))
            n2 = (1 / t2) / ((1 / t1) + (1 / t2) + (1 / t3))
            n3 = (1 / t3) / ((1 / t1) + (1 / t2) + (1 / t3))

        # If the potential service time is smaller than `1e-6`
        else:
            # Each queue will have 1/3 ratio
            n1 = 1.0 / 3
            n2 = 1.0 / 3
            n3 = 1.0 / 3

        # Calculates the number of agents that belongs to each queue
        q1 = int(n1 * n_agents)
        q2 = int(n2 * n_agents)
        q3 = int(n3 * n_agents)
        
        return q1, q2, q3

    def _update(self, agents, best_agent, function, iteration, n_iterations):
        """Method that wraps the update pipeline over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Calculates the `beta` coefficient
        beta = np.exp(np.log(1 / iteration) * np.sqrt(iteration / n_iterations))

        # Performs the first business phase
        self._business_one(agents, function, beta)

        # Performs the second business phase
        self._business_two(agents, function)

        # Performs the third business phase
        self._business_three(agents, function)

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
                self._update(space.agents, space.best_agent,
                             function, iteration, n_iterations)

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

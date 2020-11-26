"""Salp Swarm Algorithm.
"""

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class SSA(Optimizer):
    """A SSA class, inherited from Optimizer.

    This is the designed class to define SSA-related
    variables and methods.

    References:
        S. Mirjalili et al. Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems.
        Advances in Engineering Software (2017).

    """

    def __init__(self, algorithm='SSA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> SSA.')

        # Override its parent class with the receiving hyperparams
        super(SSA, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    def _update(self, agents, best_agent, iteration, n_iterations):
        """Method that wraps the Salp Swarm Algorithm over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Calculates the `c1` coefficient (Eq. 3.2)
        c1 = 2 * np.exp(-(4 * iteration / n_iterations) ** 2)

        # Iterates through every agent
        for i, _ in enumerate(agents):
            # Checks if it is the first agent
            if i == 0:
                # Iterates through every decision variable
                for j, (lb, ub) in enumerate(zip(agents[i].lb, agents[i].ub)):
                    # Generates two uniform random numbers
                    c2 = r.generate_uniform_random_number()
                    c3 = r.generate_uniform_random_number()

                    # Checks if random number is smaller than 0.5
                    if c3 < 0.5:
                        # Updates the leading salp position (Eq. 3.1 - part 1)
                        agents[i].position[j] = best_agent.position[j] + c1 * ((ub - lb) * c2 + lb)

                    # If random number is bigger or equal to 0.5
                    else:
                        # Updates the leading salp position (Eq. 3.1 - part 2)
                        agents[i].position[j] = best_agent.position[j] - c1 * ((ub - lb) * c2 + lb)

            # If it is not the first agent
            else:
                # Updates the follower salp position (Eq. 3.4)
                agents[i].position = 0.5 * (agents[i].position + agents[i-1].position)

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

"""Darcy Optimization Algorithm.
"""

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as rnd
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class DOA(Optimizer):
    """A DOA class, inherited from Optimizer.

    This is the designed class to define DOA-related
    variables and methods.

    References:
        F. Demir et al. A survival classification method for hepatocellular carcinoma patients
        with chaotic Darcy optimization method based feature selection.
        Medical Hypotheses (2020).

    """

    def __init__(self, algorithm='DOA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> DOA.')

        # Override its parent class with the receiving hyperparams
        super(DOA, self).__init__(algorithm)

        # Chaos multiplier
        self.r = 1.0

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def r(self):
        """float: Chaos multiplier.

        """

        return self._r

    @r.setter
    def r(self, r):
        if not isinstance(r, (float, int)):
            raise e.TypeError('`r` should be a float or integer')
        if r < 0:
            raise e.ValueError('`r` should be >= 0')

        self._r = r

    def _chaotic_map(self, lb, ub):
        """Calculates the chaotic maps (eq. 3).

        Args:
            lb (float): Lower bound value.
            ub (float): Upper bound value.

        Returns:
            A new value for the chaotic map.

        """

        # Generates a uniform random number between variable's bounds
        r1 = rnd.generate_uniform_random_number(lb, ub)

        # Calculates the chaotic map (eq. 3)
        c_map = self.r * r1 * (1 - r1) + ((4 - self.r) * np.sin(np.pi * r1)) / 4

        return c_map

    def _update(self, agents, best_agent, chaotic_map):
        """Method that wraps global and local pollination updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            chaotic_map (np.array): Array of current chaotic maps.

        """

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Iterates through all decision variables
            for j, (lb, ub) in enumerate(zip(agent.lb, agent.ub)):
                # Generates a chaotic map
                c_map = self._chaotic_map(lb, ub)

                # Updates the agent's position (eq. 6)
                agent.position[j] += (2 * (best_agent.position[j] - agent.position[j]) / (
                    c_map - chaotic_map[i][j])) * (ub - lb) / len(agents)

                # Updates current chaotic map with newer value
                chaotic_map[i][j] = c_map

                # Checks if position has exceed the bounds
                if (agent.position[j] < lb) or (agent.position[j] > ub):
                    # If yes, replace its value with the proposed equation (eq. 7)
                    agent.position[j] = best_agent.position[j] * c_map

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

        # Instantiates an array to hold the chaotic maps
        chaotic_map = np.zeros((space.n_agents, space.n_variables))

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
                self._update(space.agents, space.best_agent, chaotic_map)

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

"""Multi-Verse Optimizer.
"""

import numpy as np
from tqdm import tqdm

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class MVO(Optimizer):
    """A MVO class, inherited from Optimizer.

    This is the designed class to define MVO-related
    variables and methods.

    References:
        S. Mirjalili, S. M. Mirjalili and A. Hatamlou.
        Multi-verse optimizer: a nature-inspired algorithm for global optimization.
        Neural Computing and Applications (2016).

    """

    def __init__(self, algorithm='MVO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Override its parent class with the receiving hyperparams
        super(MVO, self).__init__(algorithm)

        # Minimum value for the Wormhole Existence Probability
        self.WEP_min = 0.2

        # Maximum value for the Wormhole Existence Probability
        self.WEP_max = 1.0

        # Exploitation accuracy
        self.p = 6.0

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def WEP_min(self):
        """float: Minimum Wormhole Existence Probability.

        """

        return self._WEP_min

    @WEP_min.setter
    def WEP_min(self, WEP_min):
        if not isinstance(WEP_min, (float, int)):
            raise e.TypeError('`WEP_min` should be a float or integer')
        if (WEP_min < 0 or WEP_min > 1):
            raise e.ValueError('`WEP_min` should be >= 0 and < 1')

        self._WEP_min = WEP_min

    @property
    def WEP_max(self):
        """float: Maximum Wormhole Existence Probability.

        """

        return self._WEP_max

    @WEP_max.setter
    def WEP_max(self, WEP_max):
        if not isinstance(WEP_max, (float, int)):
            raise e.TypeError('`WEP_max` should be a float or integer')
        if (WEP_max < 0 or WEP_max > 1):
            raise e.ValueError('`WEP_max` should be >= 0 and < 1')
        if WEP_max < self.WEP_min:
            raise e.ValueError('`WEP_max` should be >= `WEP_min`')

        self._WEP_max = WEP_max

    @property
    def p(self):
        """float: Exploitation accuracy.

        """

        return self._p

    @p.setter
    def p(self, p):
        if not isinstance(p, (float, int)):
            raise e.TypeError('`p` should be a float or integer')
        if p < 0:
            raise e.ValueError('`p` should be >= 0')

        self._p = p

    def _update(self, agents, best_agent, function, WEP, TDR):
        """Method that wraps updates over all agents and variables (eq. 3.1-3.4).

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.
            WEP (float): Current iteration's Wormhole Existence Probability.
            TDR (floar): Current iteration's Travelling Distance Rate.

        """

        # Gathers the fitness for each individual
        fitness = [agent.fit for agent in agents]

        # Calculates the norm of the fitness
        norm = np.linalg.norm(fitness)

        # Normalizes every individual's fitness
        norm_fitness = fitness / norm

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # For every decision variable
            for j in range(agent.n_variables):
                # Generates a uniform random number
                r1 = r.generate_uniform_random_number()

                # If random number is smaller than agent's normalized fitness
                if r1 < norm_fitness[i]:
                    # Selects a white hole through weight-based roulette wheel
                    white_hole = g.weighted_wheel_selection(norm_fitness)

                    # Gathers current agent's position as white hole's position
                    agent.position[j] = agents[white_hole].position[j]

                # Generates a second uniform random number
                r2 = r.generate_uniform_random_number()

                # If random number is smaller than WEP
                if r2 < WEP:
                    # Generates a third uniform random number
                    r3 = r.generate_uniform_random_number()

                    # Calculates the width between lower and upper bounds
                    width = r.generate_uniform_random_number(agent.lb[j], agent.ub[j])

                    # If random number is smaller than 0.5
                    if r3 < 0.5:
                        # Updates the agent's position with `+`
                        agent.position[j] = best_agent.position[j] + TDR * width

                    # If not
                    else:
                        # Updates the agent's position with `-`
                        agent.position[j] = best_agent.position[j] - TDR * width

            # Clips the agent limits
            agent.clip_limits()

            # Calculates its fitness
            agent.fit = function(agent.position)

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

                # Calculates the Wormhole Existence Probability
                WEP = self.WEP_min + (t + 1) * ((self.WEP_max - self.WEP_min) / space.n_iterations)

                # Calculates the Travelling Distance Rate
                TDR = 1 - ((t + 1) ** (1 / self.p) / space.n_iterations ** (1 / self.p))

                # Updating agents
                self._update(space.agents, space.best_agent, function, WEP, TDR)

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

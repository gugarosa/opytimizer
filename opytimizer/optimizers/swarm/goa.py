"""Grasshopper Optimization Algorithm.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.general as g
import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as log
from opytimizer.core.optimizer import Optimizer

logger = log.get_logger(__name__)


class GOA(Optimizer):
    """A GOA class, inherited from Optimizer.

    This is the designed class to define GOA-related
    variables and methods.

    References:
        S. Saremi, S. Mirjalili and A. Lewis. Grasshopper Optimisation Algorithm: Theory and application.
        Advances in Engineering Software (2017).

    """

    def __init__(self, algorithm='GOA', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> GOA.')

        # Override its parent class with the receiving hyperparams
        super(GOA, self).__init__(algorithm)

        # Minimum comfort zone
        self.c_min = 0.00001

        # Maximum comfort zone
        self.c_max = 1

        # Intensity of attraction
        self.f = 0.5

        # Attractive length scale
        self.l = 1.5

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def c_min(self):
        """float: Minimum comfort zone.

        """

        return self._c_min

    @c_min.setter
    def c_min(self, c_min):
        if not isinstance(c_min, (float, int)):
            raise e.TypeError('`c_min` should be a float or integer')
        if c_min < 0:
            raise e.ValueError('`c_min` should be >= 0')

        self._c_min = c_min

    @property
    def c_max(self):
        """float: Maximum comfort zone.

        """

        return self._c_max

    @c_max.setter
    def c_max(self, c_max):
        if not isinstance(c_max, (float, int)):
            raise e.TypeError('`c_max` should be a float or integer')
        if c_max < self.c_min:
            raise e.ValueError('`c_max` should be >= `c_min`')

        self._c_max = c_max

    @property
    def f(self):
        """float: Intensity of attraction.

        """

        return self._f

    @f.setter
    def f(self, f):
        if not isinstance(f, (float, int)):
            raise e.TypeError('`f` should be a float or integer')
        if f < 0:
            raise e.ValueError('`f` should be >= 0')

        self._f = f

    @property
    def l(self):
        """float: Attractive length scale.

        """

        return self._l

    @l.setter
    def l(self, l):
        if not isinstance(l, (float, int)):
            raise e.TypeError('`l` should be a float or integer')
        if l < 0:
            raise e.ValueError('`l` should be >= 0')

        self._l = l

    def _social_force(self, r):
        """Calculates the social force based on an input value.

        Args:
            r (np.array): Array of values.

        Returns:
            The social force based on the input value.

        """

        # Calculates the social force (Eq. 2.3)
        s = self.f * np.exp(-r / self.l) - np.exp(-r)

        return s

    def _update(self, agents, best_agent, function, iteration, n_iterations):
        """Method that wraps the Grasshopper Optimization Algorithm over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            iteration (int): Current iteration.
            n_iterations (int): Maximum number of iterations.

        """

        # Calculates the comfort coefficient (Eq. 2.8)
        comfort = self.c_max - iteration * ((self.c_max - self.c_min) / n_iterations)

        # We copy a temporary list for iterating purposes
        temp_agents = copy.deepcopy(agents)

        # Iterating through 'i' agents
        for agent in agents:
            # Initializes the total comfort as zero
            total_comfort = np.zeros((agent.n_variables, agent.n_dimensions))

            # Iterating through 'j' agents
            for temp in temp_agents:
                # Distance is calculated by an euclidean distance between 'i' and 'j'
                distance = g.euclidean_distance(agent.position, temp.position)

                # Calculates the unitary vector
                unit = (temp.position - agent.position) / (distance + c.EPSILON)

                # Calculates the social force between agents
                s = self._social_force(2 + np.fmod(distance, 2))

                # Expands the upper and lower bounds
                ub = np.expand_dims(agent.ub, -1)
                lb = np.expand_dims(agent.lb, -1)

                # Sums the current comfort to the total one
                total_comfort += comfort * ((ub - lb) / 2) * s * unit

            # Updates the agent's position (Eq. 2.7)
            agent.position = comfort * total_comfort + best_agent.position

            # Checks the agent's limits
            agent.clip_limits()

            # Evaluates the new agent's position
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

                # Updating agents
                self._update(space.agents, space.best_agent,
                             function, t, space.n_iterations)

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

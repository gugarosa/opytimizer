"""Simplified Swarm Optimization.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.decorator as d
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class SSO(Optimizer):
    """A SSO class, inherited from Optimizer.

    This is the designed class to define SSO-related
    variables and methods.

    References:
        C. Bae et al. A new simplified swarm optimization (SSO) using exchange local search scheme.
        International Journal of Innovative Computing, Information and Control (2012).

    """

    def __init__(self, algorithm='SSO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> SSO.')

        # Override its parent class with the receiving hyperparams
        super(SSO, self).__init__(algorithm)

        # Weighing constant
        self.C_w = 0.1

        # Local constant
        self.C_p = 0.4

        # Global constant
        self.C_g = 0.9

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def C_w(self):
        """float: Weighing constant.

        """

        return self._C_w

    @C_w.setter
    def C_w(self, C_w):
        if not isinstance(C_w, (float, int)):
            raise e.TypeError('`C_w` should be a float or integer')
        if C_w < 0 or C_w > 1:
            raise e.ValueError('`C_w` should be between 0 and 1')

        self._C_w = C_w

    @property
    def C_p(self):
        """float: Local constant.

        """

        return self._C_p

    @C_p.setter
    def C_p(self, C_p):
        if not isinstance(C_p, (float, int)):
            raise e.TypeError('`C_p` should be a float or integer')
        if C_p < self.C_w:
            raise e.ValueError('`C_p` should be equal or greater than `C_w`')

        self._C_p = C_p

    @property
    def C_g(self):
        """float: Global constant.

        """

        return self._C_g

    @C_g.setter
    def C_g(self, C_g):
        if not isinstance(C_g, (float, int)):
            raise e.TypeError('`C_g` should be a float or integer')
        if C_g < self.C_p:
            raise e.ValueError('`C_g` should be equal or greater than `C_p`')

        self._C_g = C_g

    def _update(self, agents, best_agent, local_position):
        """Method that wraps velocity and position updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            local_position (np.array): Array of local best posisitons.

        """

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Iterates through every decision variable
            for j in range(agent.n_variables):
                # Generates a uniform random number
                r1 = r.generate_uniform_random_number()

                # If random number is smaller than `C_w`
                if r1 < self.C_w:
                    # Ignores the position update
                    pass

                # If random number is between `C_w` and `C_p`
                elif r1 < self.C_p:
                    # Updates agent's position with its local position
                    agent.position[j] = local_position[i][j]

                # If random number is between `C_p` and `C_g`
                elif r1 < self.C_g:
                    # Updates agent's position with best position
                    agent.position[j] = best_agent.position[j]

                # If random number is greater than `C_g`
                else:
                    # Updates agent's position with random number
                    agent.position[j] = r.generate_uniform_random_number(size=agent.n_dimensions)

    @d.pre_evaluation
    def _evaluate(self, space, function, local_position):
        """Evaluates the search space according to the objective function.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            local_position (np.array): Array of local best posisitons.

        """

        # Iterate through all agents
        for i, agent in enumerate(space.agents):
            # Calculate the fitness value of current agent
            fit = function(agent.position)

            # If fitness is better than agent's best fit
            if fit < agent.fit:
                # Updates its current fitness to the newer one
                agent.fit = fit

                # Also updates the local best position to current's agent position
                local_position[i] = copy.deepcopy(agent.position)

            # If agent's fitness is better than global fitness
            if agent.fit < space.best_agent.fit:
                # Makes a deep copy of agent's local best position to the best agent
                space.best_agent.position = copy.deepcopy(local_position[i])

                # Makes a deep copy of current agent fitness to the best agent
                space.best_agent.fit = copy.deepcopy(agent.fit)

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

        # Instanciating array of local positions
        local_position = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

        # Initial search space evaluation
        self._evaluate(space, function, local_position, hook=pre_evaluation)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.file(f'Iteration {t+1}/{space.n_iterations}')

                # Updating agents
                self._update(space.agents, space.best_agent, local_position)

                # Checking if agents meet the bounds limits
                space.clip_limits()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, local_position, hook=pre_evaluation)

                # Every iteration, we need to dump agents, local positions and best agent
                history.dump(agents=space.agents,
                             local=local_position,
                             best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.file(f'Fitness: {space.best_agent.fit}')
                logger.file(f'Position: {space.best_agent.position}')

        return history

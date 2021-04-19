"""Aquila Optimizer.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.distribution as d
import opytimizer.math.random as rnd
import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class AO(Optimizer):
    """An AO class, inherited from Optimizer.

    This is the designed class to define AO-related
    variables and methods.

    References:
        L. Abualigah et al. Aquila Optimizer: A novel meta-heuristic optimization Algorithm.
        Computers & Industrial Engineering (2021).

    """

    def __init__(self, algorithm='AO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> AO.')

        # Override its parent class with the receiving hyperparams
        super(AO, self).__init__(algorithm)

        #
        self.alpha = 0.1

        #
        self.delta = 0.1

        #
        self.r1 = 10

        #
        self.U = 0.00565

        #
        self.w = 0.005

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    def _update(self, agents, best_agent, function, iteration, n_iterations):
        """Method that wraps Aquila Optimizer over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            iteration (int): Current iteration value.
            n_iterations (int): Maximum number of iterations.

        """

        # Averages the population's position
        average = np.mean([agent.position for agent in agents], axis=0)

        # Iterates through all agents
        for agent in agents:
            # Makes a deepcopy of current agent
            a = copy.deepcopy(agent)

            # Generates a random number
            r1 = rnd.generate_uniform_random_number()

            # If current iteration is smaller than 2/3 of maximum iterations
            if iteration <= ((2 / 3) * n_iterations):
                # If random number is smaller or equal to 0.5
                if r1 <= 0.5:
                    # (Eq. 3)
                    a.position = best_agent.position * (1 - (iteration / n_iterations)) + (average - best_agent.position * r1)

                # If random number is bigger than 0.5
                else:
                    #
                    levy = d.generate_levy_distribution(size=(agent.n_variables, agent.n_dimensions))

                    #
                    idx = rnd.generate_integer_random_number(high=len(agents))

                    #
                    D = np.linspace(1, agent.n_variables, agent.n_variables)
                    D = np.repeat(np.expand_dims(D, -1), agent.n_dimensions, axis=1)
                    theta = -self.w * D + (3 / 2) * np.pi
                    r = self.r1 + self.U * D

                    #
                    x = r * np.sin(theta)
                    y = r * np.cos(theta)

                    # print(a.position.shape, best_agent.position.shape, levy.shape, y.shape, x.shape, r1.shape)

                    # (Eq. 5)
                    a.position = best_agent.position * levy + agents[idx].position + (y - x) * r1

                    # print(a.position.shape)

            # If current iteration is bigger than 2/3 of maximum iterations
            else:
                # If random number is smaller or equal to 0.5
                if r1 <= 0.5:
                    #
                    lb = np.expand_dims(agent.lb, -1)
                    ub = np.expand_dims(agent.ub, -1)
                    # (Eq. 13)
                    a.position = (best_agent.position - average) * self.alpha - r1 + ((ub - lb) * r1 + lb) * self.delta

                # If random number is bigger than 0.5
                else:
                    levy = d.generate_levy_distribution(size=(agent.n_variables, agent.n_dimensions))
                    #
                    qf = iteration ** ((2 * r1 - 1) / (1 - n_iterations) ** 2)

                    #
                    G1 = 2 * r1 - 1
                    G2 = 2 * (1 - (iteration / n_iterations))

                    # (Eq. 14)
                    a.position = qf * best_agent.position - (G1 * a.position * r1) - G2 * levy + r1 * G1

            # Check agent limits
            a.clip_limits()

            # Calculates the fitness for the temporary position
            a.fit = function(a.position)

            # If new fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Copy its position and fitness to the agent
                agent.position = copy.deepcopy(a.position)
                agent.fit = copy.deepcopy(a.fit)


    def run(self, space, function, store_best_only=False, pre_evaluate=None):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (bool): If True, only the best agent of each iteration is stored in History.
            pre_evaluate (callable): This function is executed before evaluating the function being optimized.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Initial search space evaluation
        self._evaluate(space, function, hook=pre_evaluate)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.to_file(f'Iteration {t+1}/{space.n_iterations}')

                # Updating agents
                self._update(space.agents, space.best_agent, function, t, space.n_iterations)

                # Checking if agents meet the bounds limits
                space.clip_limits()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, hook=pre_evaluate)

                # Every iteration, we need to dump agents and best agent
                history.dump(agents=space.agents, best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.to_file(f'Fitness: {space.best_agent.fit}')
                logger.to_file(f'Position: {space.best_agent.position}')

        return history

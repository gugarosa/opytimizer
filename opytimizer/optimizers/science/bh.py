import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer
from opytimizer.utils import constants

logger = l.get_logger(__name__)


class BH(Optimizer):
    """A BH class, inherited from Optimizer.

    This is the designed class to define BH-related
    variables and methods.

    References:
        A. Hatamlou. Black hole: A new heuristic optimization approach for data clustering.
        Information Sciences (2013).

    """

    def __init__(self, algorithm='BH', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> BH.')

        # Override its parent class with the receiving hyperparams
        super(BH, self).__init__(algorithm)

        # Now, we need to build this class up
        self._build()

        logger.info('Class overrided.')

    def _build(self):
        """This method serves as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        """

        logger.debug('Running private method: build().')

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(f'Algorithm: {self.algorithm} | Built: {self.built}.')

    def _update_position(self, agents, best_agent, function):
        """It updates every star position and calculates their event's horizon cost.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.

        Returns:
            The cost of the event horizon.

        """

        # Event's horizon cost
        cost = 0

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # Generate an uniform random number
            r1 = r.generate_uniform_random_number()

            # Updates agent's position according to Equation 3
            agent.position += r1 * (best_agent.position - agent.position)

            # Checking agents limits
            agent.clip_limits()

            # Evaluates agent
            agent.fit = function(agent.position)

            # If new agent's fitness is better than best
            if agent.fit < best_agent.fit:
                # Swap their positions
                agent.position, best_agent.position = best_agent.position, agent.position

                # Also swap their fitness
                agent.fit, best_agent.fit = best_agent.fit, agent.fit

            # Increment the cost with current agent's fitness
            cost += agent.fit

        return cost

    def _event_horizon(self, agents, best_agent, cost):
        """It calculates the stars' crossing an event horizon.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            cost (float): The event's horizon cost.

        """

        # Calculates the radius of the event horizon
        radius = best_agent.fit / max(cost, constants.EPSILON)

        # Iterate through every agent
        for i, agent in enumerate(agents):
            # Calculates distance between star and black hole
            distance = (np.linalg.norm(best_agent.position - agent.position))

            # If distance is smaller than horizon's radius
            if distance < radius:
                # Generates a new random star
                for j, (lb, ub) in enumerate(zip(agent.lb, agent.ub)):
                    # For each decision variable, we generate uniform random numbers
                    agent.position[j] = r.generate_uniform_random_number(
                        lb, ub, size=agent.n_dimensions)

    def _update(self, agents, best_agent, function):
        """Method that wraps the update pipeline over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.

        """

        # Updates stars position and calculate their cost (Equation 3)
        cost = self._update_position(agents, best_agent, function)

        # Performs the Event Horizon (Equation 4)
        self._event_horizon(agents, best_agent, cost)

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

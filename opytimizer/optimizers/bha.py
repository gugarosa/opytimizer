import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class BHA(Optimizer):
    """A BHA class, inherited from Optimizer.

    This is the designed class to define BHA-related
    variables and methods.

    References:
        A. Hatamlou. Black hole: A new heuristic optimization approach for data clustering. Information Sciences (2013).

    """

    def __init__(self, algorithm='BHA', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> BHA.')

        # Override its parent class with the receiving hyperparams
        super(BHA, self).__init__(algorithm=algorithm)

        # Now, we need to build this class up
        self._build()

        logger.info('Class overrided.')

    def _build(self):
        """This method serves as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.debug('Running private method: build().')

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm}.')

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
            r1 = r.generate_uniform_random_number(0, 1)

            # Updates agent's position according to Equation 3
            agent.position += r1 * (best_agent.position - agent.position)

            # Checking agents limits
            agent.check_limits()

            # Evaluates agent
            agent.fit = function.pointer(agent.position)

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
        radius = best_agent.fit / cost

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

    def run(self, space, function):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Initial search space evaluation
        self._evaluate(space, function)

        # We will define a History object for further dumping
        history = h.History()

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents, space.best_agent, function)

            # Checking if agents meets the bounds limits
            space.check_limits()

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            # Every iteration, we need to dump agents, best agent and best agent's index
            history.dump(agents=space.agents, best=space.best_agent, best_index=space.best_index)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

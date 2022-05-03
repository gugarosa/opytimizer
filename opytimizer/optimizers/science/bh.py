"""Black Hole.
"""

import numpy as np

import opytimizer.math.random as r
from opytimizer.core import Optimizer
from opytimizer.utils import constant, logging

logger = logging.get_logger(__name__)


class BH(Optimizer):
    """A BH class, inherited from Optimizer.

    This is the designed class to define BH-related
    variables and methods.

    References:
        A. Hatamlou. Black hole: A new heuristic optimization approach for data clustering.
        Information Sciences (2013).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> BH.")

        # Overrides its parent class with the receiving params
        super(BH, self).__init__()

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    def _update_position(self, agents, best_agent, function):
        """It updates every star position and calculates their event's horizon cost (eq. 3).

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.

        Returns:
            The cost of the event horizon.

        """

        # Event's horizon cost
        cost = 0

        # Iterates through all agents
        for agent in agents:
            # Generate an uniform random number
            r1 = r.generate_uniform_random_number()

            # Updates agent's position
            agent.position += r1 * (best_agent.position - agent.position)

            # Checks agents limits
            agent.clip_by_bound()

            # Evaluates agent
            agent.fit = function(agent.position)

            # If new agent's fitness is better than best
            if agent.fit < best_agent.fit:
                # Swap their positions and their fitness
                agent.position, best_agent.position = (
                    best_agent.position,
                    agent.position,
                )
                agent.fit, best_agent.fit = best_agent.fit, agent.fit

            # Increment the cost with current agent's fitness
            cost += agent.fit

        return cost

    def _event_horizon(self, agents, best_agent, cost):
        """It calculates the stars' crossing an event horizon (eq. 4).

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            cost (float): The event's horizon cost.

        """

        # Calculates the radius of the event horizon
        radius = best_agent.fit / max(cost, constant.EPSILON)

        # Iterate through every agent
        for agent in agents:
            # Calculates distance between star and black hole
            distance = np.linalg.norm(best_agent.position - agent.position)

            # If distance is smaller than horizon's radius
            if distance < radius:
                # Fills agent with new random positions
                agent.fill_with_uniform()

    def update(self, space, function):
        """Wraps Black Hole over all agents and variables.

        Args:
            space (Space): Space containing agents and update-related information.
            function (Function): A Function object that will be used as the objective function.

        """

        # Updates stars position and calculate their cost (eq. 3)
        cost = self._update_position(space.agents, space.best_agent, function)

        # Performs the Event Horizon (eq. 4)
        self._event_horizon(space.agents, space.best_agent, cost)

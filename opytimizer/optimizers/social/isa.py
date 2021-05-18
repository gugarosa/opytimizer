"""Interactive Search Algorithm.
"""

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class ISA(Optimizer):
    """An ISA class, inherited from Optimizer.

    This is the designed class to define ISA-related
    variables and methods.

    References:
        A. Mortazavi, V. Toğan and A. Nuhoğlu.
        Interactive search algorithm: A new hybrid metaheuristic optimization algorithm.
        Engineering Applications of Artificial Intelligence (2018).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> ISA.')

        # Overrides its parent class with the receiving params
        super(ISA, self).__init__()

        # Inertia weight
        self.w = 0.7

        # Tendency factor
        self.tau = 0.3

        # Builds the class
        self.build(params)

        logger.info('Class overrided.')

    @property
    def w(self):
        """float: Inertia weight.

        """

        return self._w

    @w.setter
    def w(self, w):
        if not isinstance(w, (float, int)):
            raise e.TypeError('`w` should be a float or integer')
        if w < 0:
            raise e.ValueError('`w` should be >= 0')

        self._w = w

    @property
    def tau(self):
        """float: Tendency factor.

        """

        return self._tau

    @tau.setter
    def tau(self, tau):
        if not isinstance(tau, (float, int)):
            raise e.TypeError('`tau` should be a float or integer')
        if tau < 0:
            raise e.ValueError('`tau` should be >= 0')

        self._tau = tau

    @property
    def local_position(self):
        """np.array: Array of velocities.

        """

        return self._local_position

    @local_position.setter
    def local_position(self, local_position):
        if not isinstance(local_position, np.ndarray):
            raise e.TypeError('`local_position` should be a numpy array')

        self._local_position = local_position

    @property
    def velocity(self):
        """np.array: Array of velocities.

        """

        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        if not isinstance(velocity, np.ndarray):
            raise e.TypeError('`velocity` should be a numpy array')

        self._velocity = velocity

    def compile(self, space):
        """Compiles additional information that is used by this optimizer.

        Args:
            space (Space): A Space object containing meta-information.

        """

        # Arrays of local positions and velocities
        self.local_position = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))
        self.velocity = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

    def evaluate(self, space, function):
        """Evaluates the search space according to the objective function.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Calculates the fitness value of current agent
            fit = function(agent.position)

            # If fitness is better than agent's best fit
            if fit < agent.fit:
                # Updates its current fitness to the newer one
                agent.fit = fit

                # Also updates the local best position to current's agent position
                self.local_position[i] = copy.deepcopy(agent.position)

            # If agent's fitness is better than global fitness
            if agent.fit < space.best_agent.fit:
                # Makes a deep copy of agent's local best position and fitness to the best agent
                space.best_agent.position = copy.deepcopy(self.local_position[i])
                space.best_agent.fit = copy.deepcopy(agent.fit)

    def update(self, space, function):
        """Wraps Interactive Search Algorithm over all agents and variables.

        Args:
            space (Space): Space containing agents and update-related information.
            function (Function): A Function object that will be used as the objective function.

        """

        # Sorts the agents
        space.agents.sort(key=lambda x: x.fit)

        # Gathers best and worst agents
        best, worst = space.agents[0], space.agents[-1]

        # Calculates the coefficient and weighted coefficient and weighted particle (eq. 1.2)
        coef = [(best.fit - agent.fit) / (best.fit - worst.fit + c.EPSILON) for agent in space.agents]
        w_coef = [c / np.sum(coef) for c in coef]

        # Calculates weighted particle position and fitness (eq. 1.1)
        w_position = np.sum([c * agent.position for c, agent in zip(w_coef, space.agents)], axis=0)
        w_fit = function(w_position)

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Generates random uniform and integer numbers
            r1 = r.generate_uniform_random_number()
            idx = r.generate_integer_random_number(high=space.n_agents, exclude_value=i)

            # If random number is bigger than tendency factor
            if r1 >= self.tau:
                # Defines acceleration coefficients
                phi3 = r.generate_uniform_random_number()
                phi2 = 2 * r.generate_uniform_random_number()
                phi1 = -(phi2 + phi3) * r.generate_uniform_random_number()
                
                # Updates the agent's velocity (eq. 6.1)
                self.velocity[i] = self.w * self.velocity[i] + \
                                   phi1 * (self.local_position[idx] - agent.position) + \
                                   phi2 * (space.best_agent.position - self.local_position[idx]) + \
                                   phi3 * (w_position - self.local_position[idx])

            # If random number is smaller than tendency factor
            else:
                # Generates another random number
                r2 = r.generate_uniform_random_number()

                # If current agent's fitness is smaller than selected agent
                if agent.fit < space.agents[idx].fit:
                    # Updates agent's velocity (eq. 6.2 - top)
                    self.velocity[i] = r2 * (agent.position - space.agents[idx].position)
                
                # If current agent's fitness is bigger than selected agent
                else:
                    # Updates agent's velocity (eq. 6.2 - bottom)
                    self.velocity[i] = r2 * (space.agents[idx].position - agent.position)

            # Updates agent's position (eq. 6.3)
            agent.position += self.velocity[i]

            # Evaluates agent's and local's fitnesses
            agent.fit = function(agent.position)
            local_fit = function(self.local_position[i])

            if w_fit < local_fit:
                self.local_position[i] = w_position

            if agent.fit < w_fit:
                self.local_position[i] = agent.position




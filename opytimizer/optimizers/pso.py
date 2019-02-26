import copy

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.common as c
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class PSO(Optimizer):
    """A PSO class, inherited from Optimizer.
    This will be the designed class to define PSO-related
    variables and methods.

    References:
        J. Kennedy, R. C. Eberhart and Y. Shi. Swarm intelligence. Artificial Intelligence (2001). 

    Properties:
        w (float): Inertia weight.
        c1 (float): Cognitive constant.
        c2 (float): Social constant.
        local_position (np.array): An array holding particle's local positions.
        velocity (np.array): An array holding particles' velocities.

    Methods:
        _build(hyperparams): Sets an external function point to a class attribute.
        _update_velocity(agent_position, best_position, local_position, current_velocity): Updates a single particle velocity (over a single variable).
        _update_position(agent_position, current_velocity): Updates a single particle position (over a single variable).
        _update(agents, best_agent, local_position, velocity): Updates the agents' position array.
        _evaluate(space, function, local_position): Evaluates the search space according to the objective function.
        run(space, function): Runs the optimization pipeline.

    """

    def __init__(self, hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): An hyperparams dictionary containing key-value
            parameters to meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> PSO.')

        # Override its parent class with the receiving hyperparams
        super(PSO, self).__init__(algorithm='PSO')

        # Inertia weight
        self._w = 0.7

        # Cognitive constant
        self._c1 = 1.7

        # Social constant
        self._c2 = 1.7

        # Particles' local positions
        self._local_position = None

        # Particles' velocities
        self._velocity = None

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def w(self):
        """Inertia weight.
        """

        return self._w

    @w.setter
    def w(self, w):
        self._w = w

    @property
    def c1(self):
        """Cognitive constant.
        """

        return self._c1

    @c1.setter
    def c1(self, c1):
        self._c1 = c1

    @property
    def c2(self):
        """Social constant.
        """

        return self._c2

    @c2.setter
    def c2(self, c2):
        self._c2 = c2

    @property
    def local_position(self):
        """Particles' local best positions.
        """

        return self._local_position

    @local_position.setter
    def local_position(self, local_position):
        self._local_position = local_position

    @property
    def velocity(self):
        """Particles' current velocities.
        """

        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = velocity

    def _build(self, hyperparams):
        """This method will serve as the object building process.
        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            hyperparams (dict): An hyperparams dictionary containing key-value
            parameters to meta-heuristics.

        """

        logger.debug('Running private method: build()')

        # We need to save the hyperparams object for faster looking up
        self.hyperparams = hyperparams

        # If one can find any hyperparam inside its object,
        # set them as the ones that will be used
        if hyperparams:
            if 'w' in hyperparams:
                self.w = hyperparams['w']
            if 'c1' in hyperparams:
                self.c1 = hyperparams['c1']
            if 'c2' in hyperparams:
                self.c2 = hyperparams['c2']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: w = {self.w}, c1 = {self.c1}, c2 = {self.c2} | Built: {self.built}')

    def _update_velocity(self, agent_position, best_position, local_position, current_velocity):
        """Updates a single particle velocity (over a single variable).

        Args:
            agent_position (float): Agent's current position.
            best_position (float): Global best position.
            local_position (float): Agent's local best position.
            current_velocity (float): Agent's current velocity.

        Returns:
            A new velocity based on PSO's paper velocity update equation.

        """

        # Generating first random number
        r1 = r.generate_uniform_random_number(0, 1)

        # Generating second random number
        r2 = r.generate_uniform_random_number(0, 1)

        # Calculates new velocity
        new_velocity = self.w * current_velocity + self.c1 * r1 * \
            (local_position - agent_position) + self.c2 * \
            r2 * (best_position - agent_position)

        return new_velocity

    def _update_position(self, agent_position, current_velocity):
        """Updates a single particle position (over a single variable).

        Args:
            agent_position (float): Agent's current position.
            current_velocity (float): Agent's current velocity.

        Returns:
            A new position based PSO's paper position update equation.

        """

        # Calculates new position
        new_position = agent_position + current_velocity

        return new_position

    def _update(self, agents, best_agent, local_position, velocity):
        """Method that wraps velocity and position updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            local_position (np.array): Array of local best posisitons.
            velocity (np.array): Array of current velocities.

        """

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # Updates current agent velocities
            velocity[i] = self._update_velocity(
                agent.position, best_agent.position, local_position[i], velocity[i])

            # Updates current agent positions
            agent.position = self._update_position(agent.position, velocity[i])
            
            # # Iterate through all variables
            # for j, _ in enumerate(agent.position):
            #     # Updates current agent and current variable velocity value
            #     velocity[i][j] = self._update_velocity(
            #         agent.position[j], best_agent.position[j], local_position[i][j], velocity[i][j])

            #     # Updates current agent and current variable position value
            #     agent.position[j] = self._update_position(
            #         agent.position[j], velocity[i][j])

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
            fit = function.pointer(agent.position)

            # If fitness is better than agent's best fit
            if fit < agent.fit:
                # Updates its current fitness to the newer one
                agent.fit = fit

                # Also updates the local best position to current's agent position
                local_position[i] = copy.deepcopy(agent.position)

            # If agent's fitness is better than global fitness
            if agent.fit < space.best_agent.fit:
                # Makes a depp copy of current agent to the best agent
                space.best_agent = copy.deepcopy(agent)

                # Also copies its position from its local best position
                space.best_agent.position = copy.deepcopy(local_position[i])

    def run(self, space, function):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        """

        # Instanciating array of local positions
        self.local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # And also an array of velocities
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # Initial search space evaluation
        self._evaluate(space, function, self.local_position)

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents, space.best_agent,
                         self.local_position, self.velocity)

            # Checking if agents meets the bounds limits
            c.check_bound_limits(space.agents, space.lb, space.ub)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function, self.local_position)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

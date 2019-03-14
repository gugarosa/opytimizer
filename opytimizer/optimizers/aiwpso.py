import copy

import numpy as np
import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class AIWPSO(Optimizer):
    """An AIWPSO class, inherited from Optimizer.

    This will be the designed class to define AIWPSO-related
    variables and methods.

    References:
        A. Nickabadi, M. M. Ebadzadeh and R. Safabakhsh. A novel particle swarm optimization algorithm with adaptive inertia weight. Applied Soft Computing (2011).

    Attributes:
        w (float): Inertia weight.
        w_min (float): Minimum inertia weight.
        w_max (float): Maximum inertia weight.
        c1 (float): Cognitive constant.
        c2 (float): Social constant.
        fitness (np.array): An array holding particles' best fitness.
        local_position (np.array): An array holding particles' local positions.
        velocity (np.array): An array holding particles' velocities.

    Methods:
        _build(hyperparams): Sets an external function point to a class attribute.
        _update_velocity(agent_position, best_position, local_position, current_velocity): Updates a single particle
            velocity (over a single variable).
        _update_position(agent_position, current_velocity): Updates a single particle
            position (over a single variable).
        _compute_success(self, agents, fitness): Computes the particles' success for updating inertia weight.
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

        logger.info('Overriding class: Optimizer -> AIWPSO.')

        # Override its parent class with the receiving hyperparams
        super(AIWPSO, self).__init__(algorithm='AIWPSO')

        # Inertia weight
        self._w = 0.7

        # Minimum inertia weight
        self._w_min = 0.1

        # Maximum inertia weight
        self._w_max = 0.9

        # Cognitive constant
        self._c1 = 1.7

        # Social constant
        self._c2 = 1.7

        # Particles' best fitness
        self._fitness = None

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
    def w_min(self):
        """Minimum inertia weight.

        """

        return self._w_min

    @w_min.setter
    def w_min(self, w_min):
        self._w_min = w_min

    @property
    def w_max(self):
        """Maximum inertia weight.

        """

        return self._w_max

    @w_max.setter
    def w_max(self, w_max):
        self._w_max = w_max

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
    def fitness(self):
        """Particles' best fitness.

        """

        return self._fitness

    @fitness.setter
    def fitness(self, fitness):
        self._fitness = fitness

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

        logger.debug('Running private method: build().')

        # We need to save the hyperparams object for faster looking up
        self.hyperparams = hyperparams

        # If one can find any hyperparam inside its object,
        # set them as the ones that will be used
        if hyperparams:
            if 'w' in hyperparams:
                self.w = hyperparams['w']
            if 'w_min' in hyperparams:
                self.w_min = hyperparams['w_min']
            if 'w_max' in hyperparams:
                self.w_max = hyperparams['w_max']
            if 'c1' in hyperparams:
                self.c1 = hyperparams['c1']
            if 'c2' in hyperparams:
                self.c2 = hyperparams['c2']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: w = {self.w}, w_min = {self.w_min}, w_max = {self.w_max}, c1 = {self.c1}, c2 = {self.c2} | Built: {self.built}.')

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
        r1 = r.generate_uniform_random_number()

        # Generating second random number
        r2 = r.generate_uniform_random_number()

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

    def _compute_success(self, agents, fitness):
        """Computes the particles' success for updating inertia weight.

        Args:
            agents (list): List of agents.
            fitness (np.array): Array of particles' best fitness.

        Returns:
            An updated value for inertia weight.

        """

        # Initial counter
        p = 0

        # Iterating through every agent
        for i, agent in enumerate(agents):
            # If current agent fitness is smaller than its best
            if agent.fit < fitness[i]:
                # Increment the counter
                p += 1

            # Replace fitness with current agent's fitness
            fitness[i] = agent.fit

        # Update inertia weight value
        w = (self.w_max - self.w_min) * (p / len(agents)) + self.w_min

        return w

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
                # Makes a deep copy of current agent fitness to the best agent
                space.best_agent.fit = copy.deepcopy(agent.fit)

                # Also copies its position from its local best position
                space.best_agent.position = copy.deepcopy(local_position[i])

    def run(self, space, function):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Instanciating array of local positions
        self.local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # An array of velocities
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # And also an array of best particle's fitness
        self.fitness = np.zeros(space.n_agents)

        # Initial search space evaluation
        self._evaluate(space, function, self.local_position)

        # Before starting the optimization process
        # We need to copy fitness values to temporary array
        for i, agent in enumerate(space.agents):
            # Copying fitness from agent's fitness
            self.fitness[i] = agent.fit

        # We will define a History object for further dumping
        history = h.History()

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents, space.best_agent,
                         self.local_position, self.velocity)

            # Checking if agents meets the bounds limits
            space.check_bound_limits(space.agents, space.lb, space.ub)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function, self.local_position)

            # Computing particle's success and updating inertia weight
            self.w = self._compute_success(space.agents, self.fitness)

            # Every iteration, we need to dump the current space agents
            history.dump(space.agents)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

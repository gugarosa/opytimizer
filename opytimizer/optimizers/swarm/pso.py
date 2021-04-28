"""Particle Swarm Optimization-based algorithms.
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


class PSO(Optimizer):
    """A PSO class, inherited from Optimizer.

    This is the designed class to define PSO-related
    variables and methods.

    References:
        J. Kennedy, R. C. Eberhart and Y. Shi. Swarm intelligence.
        Artificial Intelligence (2001).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> PSO.')

        # Overrides its parent class with the receiving params
        super(PSO, self).__init__()

        # Inertia weight
        self.w = 0.7

        # Cognitive constant
        self.c1 = 1.7

        # Social constant
        self.c2 = 1.7

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
    def c1(self):
        """float: Cognitive constant.

        """

        return self._c1

    @c1.setter
    def c1(self, c1):
        if not isinstance(c1, (float, int)):
            raise e.TypeError('`c1` should be a float or integer')
        if c1 < 0:
            raise e.ValueError('`c1` should be >= 0')

        self._c1 = c1

    @property
    def c2(self):
        """float: Social constant.

        """

        return self._c2

    @c2.setter
    def c2(self, c2):
        if not isinstance(c2, (float, int)):
            raise e.TypeError('`c2` should be a float or integer')
        if c2 < 0:
            raise e.ValueError('`c2` should be >= 0')

        self._c2 = c2

    def create_additional_vars(self, space):
        """Creates additional variables that are used by this optimizer.

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

    def update(self, space):
        """Wraps Particle Swarm Optimization over all agents and variables.

        Args:
            space (Space): Space containing agents and update-related information.

        """

        # Iterates through all agents
        for i, agent in enumerate(space.agents):
            # Generates random numbers
            r1 = r.generate_uniform_random_number()
            r2 = r.generate_uniform_random_number()

            # Updates agent's velocity (p. 294)
            self.velocity[i] = self.w * self.velocity[i] + \
                               self.c1 * r1 * (self.local_position[i] - agent.position) + \
                               self.c2 * r2 * (space.best_agent.position - agent.position)

            # Updates agent's position (p. 294)
            agent.position += self.velocity[i]


class AIWPSO(PSO):
    """An AIWPSO class, inherited from PSO.

    This is the designed class to define AIWPSO-related
    variables and methods.

    References:
        A. Nickabadi, M. M. Ebadzadeh and R. Safabakhsh.
        A novel particle swarm optimization algorithm with adaptive inertia weight.
        Applied Soft Computing (2011).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: PSO -> AIWPSO.')

        # Minimum inertia weight
        self.w_min = 0.1

        # Maximum inertia weight
        self.w_max = 0.9

        # Overrides its parent class with the receiving params
        super(AIWPSO, self).__init__(algorithm, params)

        logger.info('Class overrided.')

    @property
    def w_min(self):
        """float: Minimum inertia weight.

        """

        return self._w_min

    @w_min.setter
    def w_min(self, w_min):
        if not isinstance(w_min, (float, int)):
            raise e.TypeError('`w_min` should be a float or integer')
        if w_min < 0:
            raise e.ValueError('`w_min` should be >= 0')

        self._w_min = w_min

    @property
    def w_max(self):
        """float: Maximum inertia weight.

        """

        return self._w_max

    @w_max.setter
    def w_max(self, w_max):
        if not isinstance(w_max, (float, int)):
            raise e.TypeError('`w_max` should be a float or integer')
        if w_max < 0:
            raise e.ValueError('`w_max` should be >= 0')
        if w_max < self.w_min:
            raise e.ValueError('`w_max` should be >= `w_min`')

        self._w_max = w_max

    def _compute_success(self, agents, fitness):
        """Computes the particles' success for updating inertia weight (eq. 16).

        Args:
            agents (list): List of agents.
            fitness (np.array): Array of particles' best fitness.

        """

        # Initial counter
        p = 0

        # Iterating through every agent
        for i, agent in enumerate(agents):
            # If current agent fitness is smaller than its best
            if agent.fit < fitness[i]:
                # Increments the counter
                p += 1

            # Replaces fitness with current agent's fitness
            fitness[i] = agent.fit

        # Update inertia weight value
        self.w = (self.w_max - self.w_min) * (p / len(agents)) + self.w_min

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

        # Instanciating array of local positions and velocities
        local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))
        velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # And also an array of best particle's fitness
        fitness = np.zeros(space.n_agents)

        # Initial search space evaluation
        self._evaluate(space, function, local_position, hook=pre_evaluate)

        # Before starting the optimization process
        # We need to copy fitness values to temporary array
        for i, agent in enumerate(space.agents):
            # Copying fitness from agent's fitness
            fitness[i] = agent.fit

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.to_file(f'Iteration {t+1}/{space.n_iterations}')

                # Updates agents
                self._update(space.agents, space.best_agent,
                             local_position, velocity)

                # Checking if agents meet the bounds limits
                space.clip_by_bound()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, local_position,
                               hook=pre_evaluate)

                # Computing particle's success and updating inertia weight
                self._compute_success(space.agents, fitness)

                # Every iteration, we need to dump agents, local positions and best agent
                history.dump(agents=space.agents,
                             local=local_position,
                             best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.to_file(f'Fitness: {space.best_agent.fit}')
                logger.to_file(f'Position: {space.best_agent.position}')

        return history


class RPSO(PSO):
    """An RPSO class, inherited from Optimizer.

    This is the designed class to define RPSO-related
    variables and methods.

    References:
        M. Roder, G. H. de Rosa, L. A. Passos, A. L. D. Rossi and J. P. Papa.
        Harnessing Particle Swarm Optimization Through Relativistic Velocity.
        IEEE Congress on Evolutionary Computation (2020).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: PSO -> RPSO.')

        # Overrides its parent class with the receiving params
        super(RPSO, self).__init__(algorithm, params)

        logger.info('Class overrided.')

    def _update_velocity(self, position, best_position, local_position, max_velocity, velocity, mass):
        """Updates a single particle velocity over a single variable (eq. 11).

        Args:
            position (np.array): Agent's current position.
            best_position (np.array): Global best position.
            local_position (np.array): Agent's local best position.
            max_velocity (float): Maximum velocity of all agents.
            velocity (np.array): Agent's current velocity.
            mass (float): Agent's mass.

        Returns:
            A new velocity based on relativistic speed proposal.

        """

        # Generating random numbers
        r1 = r.generate_uniform_random_number()
        r2 = r.generate_uniform_random_number()

        # Calculating gamma parameter
        gamma = 1 / np.sqrt(1 - (max_velocity ** 2 / c.LIGHT_SPEED ** 2))

        # Calculates new velocity
        new_velocity = mass * velocity * gamma + self.c1 * r1 * (local_position - position) + self.c2 * \
            r2 * (best_position - position)

        return new_velocity

    def update(self, agents, best_agent, local_position, velocity, mass):
        """Method that wraps velocity and position updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            local_position (np.array): Array of local best posisitons.
            velocity (np.array): Array of current velocities.
            mass (np.array): Array of agents' masses.

        """

        # Calculating the maximum velocity
        max_velocity = np.max(velocity)

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Updates current agent velocities
            velocity[i] = self._update_velocity(
                agent.position, best_agent.position, local_position[i], max_velocity, velocity[i], mass[i])

            # Updates current agent positions
            agent.position = self._update_position(agent.position, velocity[i])

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

        # Instanciating array of local positions, velocities and masses
        local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))
        velocity = np.ones(
            (space.n_agents, space.n_variables, space.n_dimensions))
        mass = r.generate_uniform_random_number(
            size=(space.n_agents, space.n_variables, space.n_dimensions))

        # Initial search space evaluation
        self._evaluate(space, function, local_position, hook=pre_evaluate)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.to_file(f'Iteration {t+1}/{space.n_iterations}')

                # Updates agents
                self._update(space.agents, space.best_agent,
                             local_position, velocity, mass)

                # Checking if agents meet the bounds limits
                space.clip_by_bound()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, local_position,
                               hook=pre_evaluate)

                # Every iteration, we need to dump agents, local positions and best agent
                history.dump(agents=space.agents,
                             local=local_position,
                             best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.to_file(f'Fitness: {space.best_agent.fit}')
                logger.to_file(f'Position: {space.best_agent.position}')

        return history


class SAVPSO(PSO):
    """An SAVPSO class, inherited from Optimizer.

    This is the designed class to define SAVPSO-related
    variables and methods.

    References:
        H. Lu and W. Chen.
        Self-adaptive velocity particle swarm optimization for solving constrained optimization problems.
        Journal of global optimization (2008).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: PSO -> SAVPSO.')

        # Overrides its parent class with the receiving params
        super(SAVPSO, self).__init__(algorithm, params)

        logger.info('Class overrided.')

    def _update_velocity(self, position, best_position, local_position, selected_position, velocity):
        """Updates a single particle velocity (eq. 8).

        Args:
            position (np.array): Agent's current position.
            best_position (np.array): Global best position.
            local_position (np.array): Agent's local best position.
            selected_position (np.array): Selected agent's position.
            velocity (np.array): Agent's current velocity.

        Returns:
            A new velocity based on self-adaptive proposal.

        """

        # Generating a random number
        r1 = r.generate_uniform_random_number()

        # Calculates new velocity
        new_velocity = self.w * np.fabs(selected_position - local_position) * np.sign(velocity) + r1 * (
            local_position - position) + (1 - r1) * (best_position - position)

        return new_velocity

    def update(self, agents, best_agent, local_position, velocity):
        """Method that wraps velocity and position updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            local_position (np.array): Array of local best posisitons.
            velocity (np.array): Array of current velocities.

        """

        # Creates an array of positions
        positions = np.zeros(
            (agents[0].position.shape[0], agents[0].position.shape[1]))

        # For every agent
        for agent in agents:
            # Sums up its position
            positions += agent.position

        # Divides by the number of agents
        positions /= len(agents)

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Generates a random index for selecting an agent
            idx = r.generate_integer_random_number(0, len(agents))

            # Updates current agent's velocity
            velocity[i] = self._update_velocity(
                agent.position, best_agent.position, local_position[i], local_position[idx], velocity[i])

            # Updates current agent's position
            agent.position = self._update_position(agent.position, velocity[i])

            # For every decision variable
            for j in range(agent.n_variables):
                # Generates a random number
                r4 = r.generate_uniform_random_number(0, 1)

                # If position is greater than upper bound
                if agent.position[j] > agent.ub[j]:
                    # Replaces its value
                    agent.position[j] = positions[j] + 1 * \
                        r4 * (agent.ub[j] - positions[j])

                # If position is smaller than lower bound
                if agent.position[j] < agent.lb[j]:
                    # Replaces its value
                    agent.position[j] = positions[j] + 1 * \
                        r4 * (agent.lb[j] - positions[j])


class VPSO(PSO):
    """An VPSO class, inherited from Optimizer.

    This is the designed class to define VPSO-related
    variables and methods.

    References:
        W.-P. Yang. Vertical particle swarm optimization algorithm and its application in soft-sensor modeling.
        International Conference on Machine Learning and Cybernetics (2007).

    """

    def __init__(self, params=None):
        """Initialization method.

        Args:
            params (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: PSO -> VPSO.')

        # Overrides its parent class with the receiving params
        super(VPSO, self).__init__(algorithm, params)

        logger.info('Class overrided.')

    def _update_velocity(self, position, best_position, local_position, velocity, v_velocity):
        """Updates a single particle velocity (eq. 3-4).

        Args:
            position (np.array): Agent's current position.
            best_position (np.array): Global best position.
            local_position (np.array): Agent's local best position.
            velocity (np.array): Agent's current velocity.
            v_velocity (np.array): Agent's current vertical velocity.

        Returns:
            A new velocity based on vertical proposal.

        """

        # Generating uniform random numbers
        r1 = r.generate_uniform_random_number()
        r2 = r.generate_uniform_random_number()

        # Calculates new velocity
        new_velocity = self.w * velocity + self.c1 * r1 * (local_position - position) + self.c2 * \
            r2 * (best_position - position)

        # Calculates new vertical velocity
        new_v_velocity = v_velocity - (np.dot(new_velocity.T, v_velocity) /
                                       np.dot(new_velocity.T, new_velocity)) * new_velocity

        return new_velocity, new_v_velocity

    def _update_position(self, position, velocity, v_velocity):
        """Updates a particle position (eq. 5).

        Args:
            position (np.array): Agent's current position.
            velocity (np.array): Agent's current velocity.
            v_velocity (np.array): Agent's current vertical velocity.

        Returns:
            A new position based on VPSO's paper position update equation.

        """

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number()

        # Calculates new position
        new_position = position + r1 * velocity + (1 - r1) * v_velocity

        return new_position

    def update(self, agents, best_agent, local_position, velocity, v_velocity):
        """Method that wraps velocity and position updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            local_position (np.array): Array of local best posisitons.
            velocity (np.array): Array of current velocities.
            v_velocity (np.array): Array of currentvertical velocities.

        """

        # Iterates through all agents
        for i, agent in enumerate(agents):
            # Updates current agent velocity
            velocity[i], v_velocity[i] = self._update_velocity(
                agent.position, best_agent.position, local_position[i], velocity[i], v_velocity[i])

            # Updates current agent positions
            agent.position = self._update_position(
                agent.position, velocity[i], v_velocity[i])

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

        # Instanciating array of local positions, velocities and vertical velocities
        local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))
        velocity = np.ones(
            (space.n_agents, space.n_variables, space.n_dimensions))
        v_velocity = np.ones(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # Initial search space evaluation
        self._evaluate(space, function, local_position, hook=pre_evaluate)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.to_file(f'Iteration {t+1}/{space.n_iterations}')

                # Updates agents
                self._update(space.agents, space.best_agent,
                             local_position, velocity, v_velocity)

                # Checking if agents meet the bounds limits
                space.clip_by_bound()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, local_position,
                               hook=pre_evaluate)

                # Every iteration, we need to dump agents, local positions and best agent
                history.dump(agents=space.agents,
                             local=local_position,
                             best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.to_file(f'Fitness: {space.best_agent.fit}')
                logger.to_file(f'Position: {space.best_agent.position}')

        return history

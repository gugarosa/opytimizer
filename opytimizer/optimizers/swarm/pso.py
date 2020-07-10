import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constants as c
import opytimizer.utils.decorator as d
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

    def __init__(self, algorithm='PSO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> PSO.')

        # Override its parent class with the receiving hyperparams
        super(PSO, self).__init__(algorithm)

        # Inertia weight
        self.w = 0.7

        # Cognitive constant
        self.c1 = 1.7

        # Social constant
        self.c2 = 1.7

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def w(self):
        """float: Inertia weight.

        """

        return self._w

    @w.setter
    def w(self, w):
        if not (isinstance(w, float) or isinstance(w, int)):
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
        if not (isinstance(c1, float) or isinstance(c1, int)):
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
        if not (isinstance(c2, float) or isinstance(c2, int)):
            raise e.TypeError('`c2` should be a float or integer')
        if c2 < 0:
            raise e.ValueError('`c2` should be >= 0')

        self._c2 = c2

    def _build(self, hyperparams):
        """This method serves as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.debug('Running private method: build().')

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
            f'Algorithm: {self.algorithm} | '
            f'Hyperparameters: w = {self.w}, c1 = {self.c1}, c2 = {self.c2} | '
            f'Built: {self.built}.')

    def _update_velocity(self, position, best_position, local_position, velocity):
        """Updates a particle velocity.

        Args:
            position (np.array): Agent's current position.
            best_position (np.array): Global best position.
            local_position (np.array): Agent's local best position.
            velocity (np.array): Agent's current velocity.

        Returns:
            A new velocity based on PSO's paper velocity update equation.

        """

        # Generating first random number
        r1 = r.generate_uniform_random_number()

        # Generating second random number
        r2 = r.generate_uniform_random_number()

        # Calculates new velocity
        new_velocity = self.w * velocity + self.c1 * r1 * \
            (local_position - position) + self.c2 * \
            r2 * (best_position - position)

        return new_velocity

    def _update_position(self, position, velocity):
        """Updates a particle position.

        Args:
            position (np.array): Agent's current position.
            velocity (np.array): Agent's current velocity.

        Returns:
            A new position based on PSO's paper position update equation.

        """

        # Calculates new position
        new_position = position + velocity

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
        local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # And also an array of velocities
        velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

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
                self._update(space.agents, space.best_agent, local_position, velocity)

                # Checking if agents meets the bounds limits
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


class AIWPSO(PSO):
    """An AIWPSO class, inherited from PSO.

    This is the designed class to define AIWPSO-related
    variables and methods.

    References:
        A. Nickabadi, M. M. Ebadzadeh and R. Safabakhsh.
        A novel particle swarm optimization algorithm with adaptive inertia weight.
        Applied Soft Computing (2011).

    """

    def __init__(self, algorithm='AIWPSO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: PSO -> AIWPSO.')

        # Override its parent class with the receiving hyperparams
        super(AIWPSO, self).__init__(algorithm, hyperparams)

        # Minimum inertia weight
        self.w_min = 0.1

        # Maximum inertia weight
        self.w_max = 0.9

        # Now, we need to re-build this class up
        self._rebuild()

        logger.info('Class overrided.')

    @property
    def w_min(self):
        """float: Minimum inertia weight.

        """

        return self._w_min

    @w_min.setter
    def w_min(self, w_min):
        if not (isinstance(w_min, float) or isinstance(w_min, int)):
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
        if not (isinstance(w_max, float) or isinstance(w_max, int)):
            raise e.TypeError('`w_max` should be a float or integer')
        if w_max < 0:
            raise e.ValueError('`w_max` should be >= 0')
        if w_max < self.w_min:
            raise e.ValueError('`w_max` should be >= `w_min`')

        self._w_max = w_max

    def _rebuild(self):
        """This method serves as the object re-building process.

        One is supposed to use this class only when defining extra hyperparameters
        that can not be inherited by its parent.

        """

        logger.debug('Running private method: rebuild().')

        # If one can find any hyperparam inside its object,
        # set them as the ones that will be used
        if self.hyperparams:
            if 'w_min' in self.hyperparams:
                self.w_min = self.hyperparams['w_min']
            if 'w_max' in self.hyperparams:
                self.w_max = self.hyperparams['w_max']

        # Logging attributes
        logger.debug(
            f'Additional hyperparameters: w_min = {self.w_min}, w_max = {self.w_max}.')

    def _compute_success(self, agents, fitness):
        """Computes the particles' success for updating inertia weight.

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
                # Increment the counter
                p += 1

            # Replace fitness with current agent's fitness
            fitness[i] = agent.fit

        # Update inertia weight value
        self.w = (self.w_max - self.w_min) * (p / len(agents)) + self.w_min

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
        local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # An array of velocities
        velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # And also an array of best particle's fitness
        fitness = np.zeros(space.n_agents)

        # Initial search space evaluation
        self._evaluate(space, function, local_position, hook=pre_evaluation)

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
                logger.file(f'Iteration {t+1}/{space.n_iterations}')

                # Updating agents
                self._update(space.agents, space.best_agent, local_position, velocity)

                # Checking if agents meets the bounds limits
                space.clip_limits()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, local_position, hook=pre_evaluation)

                # Computing particle's success and updating inertia weight
                self._compute_success(space.agents, fitness)

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


class RPSO(PSO):
    """An RPSO class, inherited from Optimizer.

    This is the designed class to define RPSO-related
    variables and methods.

    References:
        M. Roder, G. H. de Rosa, L. A. Passos, A. L. D. Rossi and J. P. Papa.
        Harnessing Particle Swarm Optimization Through Relativistic Velocity.
        To be published (2020).

    """

    def __init__(self, algorithm='RPSO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: PSO -> RPSO.')

        # Override its parent class with the receiving hyperparams
        super(RPSO, self).__init__(algorithm, hyperparams)

        logger.info('Class overrided.')

    def _update_velocity(self, position, best_position, local_position, max_velocity, velocity, mass):
        """Updates a single particle velocity (over a single variable).

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

        # Generating first random number
        r1 = r.generate_uniform_random_number()

        # Generating second random number
        r2 = r.generate_uniform_random_number()

        # Calculating gamma parameter
        gamma = 1 / np.sqrt(1 - (max_velocity ** 2 / c.LIGHT_SPEED ** 2))

        # Calculates new velocity
        new_velocity = mass * velocity * gamma + self.c1 * r1 * \
            (local_position - position) + self.c2 * \
            r2 * (best_position - position)

        return new_velocity

    def _update(self, agents, best_agent, local_position, velocity, mass):
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

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # Updates current agent velocities
            velocity[i] = self._update_velocity(
                agent.position, best_agent.position, local_position[i], max_velocity, velocity[i], mass[i])

            # Updates current agent positions
            agent.position = self._update_position(agent.position, velocity[i])

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
        local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # An array of velocities
        velocity = np.ones(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # And finally, an array of masses
        mass = r.generate_uniform_random_number(size=(space.n_agents, space.n_variables, space.n_dimensions))

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
                self._update(space.agents, space.best_agent, local_position, velocity, mass)

                # Checking if agents meets the bounds limits
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


class SAVPSO(PSO):
    """An SAVPSO class, inherited from Optimizer.

    This is the designed class to define SAVPSO-related
    variables and methods.

    References:
        H. Lu and W. Chen. Self-adaptive velocity particle swarm optimization for solving constrained optimization problems.
        Journal of global optimization (2008).

    """

    def __init__(self, algorithm='SAVPSO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: PSO -> SAVPSO.')

        # Override its parent class with the receiving hyperparams
        super(SAVPSO, self).__init__(algorithm, hyperparams)

        logger.info('Class overrided.')

    def _update_velocity(self, position, best_position, local_position, selected_position, velocity):
        """Updates a single particle velocity.

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

    def _update(self, agents, best_agent, local_position, velocity):
        """Method that wraps velocity and position updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            local_position (np.array): Array of local best posisitons.
            velocity (np.array): Array of current velocities.

        """

        # Creates an array of positions
        positions = np.zeros((agents[0].position.shape[0], agents[0].position.shape[1]))

        # For every agent
        for agent in agents:
            # Sums up its position
            positions += agent.position

        # Divides by the number of agents
        positions /= len(agents)

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # Generates a random index for selecting an agent
            idx = int(r.generate_uniform_random_number(0, len(agents)))

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
                    # Replace its value
                    agent.position[j] = positions[j] + 1 * r4 * (agent.ub[0] - positions[j])

                # If position is smaller than lower bound
                if agent.position[j] < agent.lb[j]:
                    # Replace its value
                    agent.position[j] = positions[j] + 1 * r4 * (agent.lb[j] - positions[j])


class VPSO(PSO):
    """An VPSO class, inherited from Optimizer.

    This is the designed class to define VPSO-related
    variables and methods.

    References:
        W.-P. Yang. Vertical particle swarm optimization algorithm and its application in soft-sensor modeling.
        International Conference on Machine Learning and Cybernetics (2007).

    """

    def __init__(self, algorithm='VPSO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: PSO -> VPSO.')

        # Override its parent class with the receiving hyperparams
        super(VPSO, self).__init__(algorithm, hyperparams)

        logger.info('Class overrided.')

    def _update_velocity(self, position, best_position, local_position, velocity, v_velocity):
        """Updates a single particle velocity.

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
        new_velocity = self.w * velocity + self.c1 * r1 * \
            (local_position - position) + self.c2 * \
            r2 * (best_position - position)

        # Calculates new vertical velocity
        new_v_velocity = v_velocity - \
            (np.dot(new_velocity.T, v_velocity) /
             np.dot(new_velocity.T, new_velocity)) * new_velocity

        return new_velocity, new_v_velocity

    def _update_position(self, position, velocity, v_velocity):
        """Updates a particle position.

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

    def _update(self, agents, best_agent, local_position, velocity, v_velocity):
        """Method that wraps velocity and position updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            local_position (np.array): Array of local best posisitons.
            velocity (np.array): Array of current velocities.
            v_velocity (np.array): Array of currentvertical velocities.

        """

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # Updates current agent velocity
            velocity[i], v_velocity[i] = self._update_velocity(
                agent.position, best_agent.position, local_position[i], velocity[i], v_velocity[i])

            # Updates current agent positions
            agent.position = self._update_position(agent.position, velocity[i], v_velocity[i])

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
        local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # And also an array of velocities
        velocity = np.ones(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # And also an array of vertical velocities
        v_velocity = np.ones(
            (space.n_agents, space.n_variables, space.n_dimensions))

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
                self._update(space.agents, space.best_agent,
                             local_position, velocity, v_velocity)

                # Checking if agents meets the bounds limits
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

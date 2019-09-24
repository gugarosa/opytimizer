import copy

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class GP(Optimizer):
    """A GP class, inherited from Optimizer.

    This will be the designed class to define GP-related
    variables and methods.

    References:
        ...

    """

    def __init__(self, algorithm='GP', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): A string holding optimizer's algorithm name.
            hyperparams (dict): An hyperparams dictionary containing key-value
                parameters to meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> GP.')

        # Override its parent class with the receiving hyperparams
        super(GP, self).__init__(algorithm=algorithm)

        # Probability of reproduction
        self._reproduction = 0.3

        # Probability of mutation
        self._mutation = 0.4

        # Probability of crossover
        self._crossover = 0.4 

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
        self._w = w

    @property
    def c1(self):
        """float: Cognitive constant.

        """

        return self._c1

    @c1.setter
    def c1(self, c1):
        self._c1 = c1

    @property
    def c2(self):
        """float: Social constant.

        """

        return self._c2

    @c2.setter
    def c2(self, c2):
        self._c2 = c2

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
            if 'c1' in hyperparams:
                self.c1 = hyperparams['c1']
            if 'c2' in hyperparams:
                self.c2 = hyperparams['c2']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: w = {self.w}, c1 = {self.c1}, c2 = {self.c2} | Built: {self.built}.')

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
                # Makes a deep copy of current agent's index to the space's best index
                space.best_index = i

                # Makes a deep copy of agent's local best position to the best agent
                space.best_agent.position = copy.deepcopy(local_position[i])

                # Makes a deep copy of current agent fitness to the best agent
                space.best_agent.fit = copy.deepcopy(agent.fit)

    def run(self, space, function):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

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
        self._evaluate(space, function, local_position)

        # We will define a History object for further dumping
        history = h.History()

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents, space.best_agent,
                         local_position, velocity)

            # Checking if agents meets the bounds limits
            space.check_limits()

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function, local_position)

            # Every iteration, we need to dump the current space agents
            history.dump_pso(local_position, space.agents, space.best_agent, space.best_index)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

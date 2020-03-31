import copy

import numpy as np

import opytimizer.math.random as r
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
        J. Kennedy, R. C. Eberhart and Y. Shi. Swarm intelligence. Artificial Intelligence (2001). 

    """

    def __init__(self, algorithm='PSO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> PSO.')

        # Override its parent class with the receiving hyperparams
        super(PSO, self).__init__(algorithm=algorithm)

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
            f'Algorithm: {self.algorithm} | Hyperparameters: w = {self.w}, c1 = {self.c1}, c2 = {self.c2} | Built: {self.built}.')

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
            A new position based PSO's paper position update equation.

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
                # Makes a deep copy of agent's local best position to the best agent
                space.best_agent.position = copy.deepcopy(local_position[i])

                # Makes a deep copy of current agent fitness to the best agent
                space.best_agent.fit = copy.deepcopy(agent.fit)

    def run(self, space, function, store_best_only=False, pre_evaluation_hook=None):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (bool): If True, only the best agent of each iteration is stored in History.
            pre_evaluation_hook (callable): A function that receives the optimizer, space and function
                and returns None. This function is executed before evaluating the function being optimized.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Instanciating array of local positions
        local_position = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # And also an array of velocities
        velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # Check if there is a pre-evaluation hook
        if pre_evaluation_hook:
            # Applies the hook
            pre_evaluation_hook(self, space, function)

        # Initial search space evaluation
        self._evaluate(space, function, local_position)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents, space.best_agent,
                         local_position, velocity)

            # Checking if agents meets the bounds limits
            space.clip_limits()

            # Check if there is a pre-evaluation hook
            if pre_evaluation_hook:
                # Applies the hook
                pre_evaluation_hook(self, space, function)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function, local_position)

            # Every iteration, we need to dump agents, local positions and best agent
            history.dump(agents=space.agents, local=local_position, best_agent=space.best_agent)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

import numpy as np

import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.optimizers.pso import PSO

logger = l.get_logger(__name__)


class AIWPSO(PSO):
    """An AIWPSO class, inherited from PSO.

    This is the designed class to define AIWPSO-related
    variables and methods.

    References:
        A. Nickabadi, M. M. Ebadzadeh and R. Safabakhsh. A novel particle swarm optimization algorithm with adaptive inertia weight. Applied Soft Computing (2011).

    """

    def __init__(self, algorithm='AIWPSO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: PSO -> AIWPSO.')

        # Override its parent class with the receiving hyperparams
        super(AIWPSO, self).__init__(
            algorithm=algorithm, hyperparams=hyperparams)

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

        # An array of velocities
        velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # And also an array of best particle's fitness
        fitness = np.zeros(space.n_agents)

        # Initial search space evaluation
        self._evaluate(space, function, local_position)

        # Before starting the optimization process
        # We need to copy fitness values to temporary array
        for i, agent in enumerate(space.agents):
            # Copying fitness from agent's fitness
            fitness[i] = agent.fit

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

            # Computing particle's success and updating inertia weight
            self._compute_success(space.agents, fitness)

            # Every iteration, we need to dump agents, local positions, best agent and best agent's index
            history.dump(agents=space.agents, local=local_position,
                         best=space.best_agent, best_index=space.best_index)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

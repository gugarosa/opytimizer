import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class WDO(Optimizer):
    """A WDO class, inherited from Optimizer.

    This is the designed class to define WDO-related
    variables and methods.

    References:
        Z. Bayraktar, et al. The wind driven optimization technique and its application in electromagnetics. 
        IEEE transactions on antennas and propagation (2013).

    """

    def __init__(self, algorithm='WDO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> WDO.')

        # Override its parent class with the receiving hyperparams
        super(WDO, self).__init__(algorithm)

        # Maximum velocity
        self.v_max = 0.3

        # Friction coefficient
        self.alpha = 0.8

        # Gravitational force coefficient
        self.g = 0.6

        # Coriolis force
        self.c = 1.0

        # Pressure constant
        self.RT = 1.5

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def v_max(self):
        """float: Maximum velocity.

        """

        return self._v_max

    @v_max.setter
    def v_max(self, v_max):
        if not (isinstance(v_max, float) or isinstance(v_max, int)):
            raise e.TypeError('`v_max` should be a float or integer')
        if v_max < 0:
            raise e.ValueError('`v_max` should be >= 0')

        self._v_max = v_max

    @property
    def alpha(self):
        """float: Friction coefficient.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if not (isinstance(alpha, float) or isinstance(alpha, int)):
            raise e.TypeError('`alpha` should be a float or integer')
        if alpha < 0 or alpha > 1:
            raise e.ValueError('`alpha` should be between 0 and 1')

        self._alpha = alpha

    @property
    def g(self):
        """float: Gravitational force coefficient.

        """

        return self._g

    @g.setter
    def g(self, g):
        if not (isinstance(g, float) or isinstance(g, int)):
            raise e.TypeError('`g` should be a float or integer')
        if g < 0:
            raise e.ValueError('`g` should be >= 0')

        self._g = g

    @property
    def c(self):
        """float: Coriolis force.

        """

        return self._c

    @c.setter
    def c(self, c):
        if not (isinstance(c, float) or isinstance(c, int)):
            raise e.TypeError('`c` should be a float or integer')
        if c < 0:
            raise e.ValueError('`c` should be >= 0')

        self._c = c

    @property
    def RT(self):
        """float: Pressure constant.

        """

        return self._RT

    @RT.setter
    def RT(self, RT):
        if not (isinstance(RT, float) or isinstance(RT, int)):
            raise e.TypeError('`RT` should be a float or integer')
        if RT < 0:
            raise e.ValueError('`RT` should be >= 0')

        self._RT = RT

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
            if 'v_max' in hyperparams:
                self.v_max = hyperparams['v_max']
            if 'alpha' in hyperparams:
                self.alpha = hyperparams['alpha']
            if 'g' in hyperparams:
                self.g = hyperparams['g']
            if 'c' in hyperparams:
                self.c = hyperparams['c']
            if 'RT' in hyperparams:
                self.RT = hyperparams['RT']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | '
            f'Hyperparameters: v_max = {self.v_max}, alpha = {self.alpha}, g = {self.g}, c = {self.c}, RT = {self.RT} | '
            f'Built: {self.built}.')

    def _update_velocity(self, position, best_position, velocity, alt_velocity, index):
        """Updates an agent velocity.

        Args:
            position (np.array): Agent's current position.
            best_position (np.array): Global best position.
            velocity (np.array): Agent's current velocity.
            alt_velocity (np.array): Random agent's current velocity.
            index (int): Index of current agent.

        Returns:
            A new velocity based on on WDO's paper equation 15.

        """

        # Calculates new velocity
        new_velocity = (1 - self.alpha) * velocity - self.g * position + (self.RT * np.abs(
            1 / index - 1) * (best_position - position)) + (self.c * alt_velocity / index)

        return new_velocity

    def _update_position(self, position, velocity):
        """Updates an agent position.

        Args:
            position (np.array): Agent's current position.
            velocity (np.array): Agent's current velocity.

        Returns:
            A new position based on WDO's paper equation 16.

        """

        # Calculates new position
        new_position = position + velocity

        return new_position

    def _update(self, agents, best_agent, function, velocity):
        """Method that wraps velocity and position updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            velocity (np.array): Array of current velocities.

        """

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # Generates a random index based on the number of agents
            index = int(r.generate_uniform_random_number(0, len(velocity)))

            # Updating velocity
            velocity[i] = self._update_velocity(
                agent.position, best_agent.position, velocity[i], velocity[index], i + 1)

            # Clips the velocity values between (-v_max, v_max)
            velocity = np.clip(velocity, -self.v_max, self.v_max)

            # Updating agent's position
            agent.position = self._update_position(agent.position, velocity[i])

            # Checks agent limits
            agent.clip_limits()

            # Evaluates agent
            agent.fit = function(agent.position)

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

        # Instanciating array of velocities
        velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # Initial search space evaluation
        self._evaluate(space, function, hook=pre_evaluation)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.file(f'Iteration {t+1}/{space.n_iterations}')

                # Updating agents
                self._update(space.agents, space.best_agent, function, velocity)

                # Checking if agents meets the bounds limits
                space.clip_limits()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, hook=pre_evaluation)

                # Every iteration, we need to dump agents and best agent
                history.dump(agents=space.agents, best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.file(f'Fitness: {space.best_agent.fit}')
                logger.file(f'Position: {space.best_agent.position}')

        return history

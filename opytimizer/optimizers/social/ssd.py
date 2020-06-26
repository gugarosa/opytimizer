import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.decorator as d
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class SSD(Optimizer):
    """An SSD class, inherited from Optimizer.

    This is the designed class to define SSD-related
    variables and methods.

    References:
        A. Tharwat and T. Gabel.
        Parameters optimization of support vector machines for imbalanced data using social ski driver algorithm.
        Neural Computing and Applications (2019).

    """

    def __init__(self, algorithm='SSD', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> SSD.')

        # Override its parent class with the receiving hyperparams
        super(SSD, self).__init__(algorithm)

        # Exploration parameter
        self.c = 2.0

        # Decay rate
        self.decay = 0.99

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def c(self):
        """float: Exploration parameter.

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
    def decay(self):
        """float: Decay rate.

        """

        return self._decay

    @decay.setter
    def decay(self, decay):
        if not (isinstance(decay, float) or isinstance(decay, int)):
            raise e.TypeError('`decay` should be a float or integer')
        if decay < 0 or decay > 1:
            raise e.ValueError('`decay` should be between 0 and 1')
        self._decay = decay

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
            if 'c' in hyperparams:
                self.c = hyperparams['c']
            if 'decay' in hyperparams:
                self.decay = hyperparams['decay']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: c = {self.c}, decay = {self.decay} | '
            f'Built: {self.built}.')

    def _mean_global_solution(self, alpha, beta, gamma):
        """Calculates the mean global solution.

        Args:
            alpha (np.array): 1st agent's current position.
            beta (np.array): 2nd agent's current position.
            gamma (np.array): 3rd agent's current position.

        Returns:
            Mean global solution based on SSD's paper mean global solution equation.

        """

        # Calculates the mean global solution
        mean = (alpha + beta + gamma) / 3

        return mean

    def _update_position(self, position, velocity):
        """Updates a particle position.

        Args:
            position (np.array): Agent's current position.
            velocity (np.array): Agent's current velocity.

        Returns:
            A new position based on SSD's paper position update equation.

        """

        # Calculates new position
        new_position = position + velocity

        return new_position

    def _update_velocity(self, position, mean, local_position, velocity):
        """Updates a particle velocity.

        Args:
            position (np.array): Agent's current position.
            mean (np.array): Mean global best position.
            local_position (np.array): Agent's local best position.
            velocity (np.array): Agent's current velocity.

        Returns:
            A new velocity based on SSD's paper velocity update equation.

        """

        # Generating first random number
        r1 = r.generate_uniform_random_number()

        # Generating second random number
        r2 = r.generate_uniform_random_number()

        # If random number is smaller than or equal to 0.5
        if r2 <= 0.5:
            # Updates its velocity based on sine wave
            new_velocity = self.c * \
                np.sin(r1) * (local_position - position) + \
                np.sin(r1) * (mean - position)

        # If random number is bigger than 0.5
        else:
            # Updates its velocity based on cosine wave
            new_velocity = self.c * \
                np.cos(r1) * (local_position - position) + \
                np.cos(r1) * (mean - position)

        return new_velocity

    def _update(self, agents, function, local_position, velocity):
        """Method that wraps mean global solution, position and velocity updates over all agents and variables.

        Args:
            agents (list): List of agents.
            function (Function): A Function object that will be used as the objective function.
            local_position (np.array): Array of local best posisitons.
            velocity (np.array): Array of current velocities.

        """

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # Calculates the new fitness
            fit = function(agent.position)

            # If new fitness is better than agent's fitness
            if fit < agent.fit:
                # Updates its current fitness to the newer one
                agent.fit = fit

                # Also updates the local best position to current agent's position
                local_position[i] = copy.deepcopy(agent.position)

            # Sorting agents
            agents.sort(key=lambda x: x.fit)

            # Calculates the mean global solution
            mean = self._mean_global_solution(
                agents[0].position, agents[1].position, agents[2].position)

            # Updates current agent positions
            agent.position = self._update_position(agent.position, velocity[i])

            # Checking agent limits
            agent.clip_limits()

            # Updates current agent velocities
            velocity[i] = self._update_velocity(
                agent.position, mean, local_position[i], velocity[i])

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
        velocity = r.generate_uniform_random_number(
            size=(space.n_agents, space.n_variables, space.n_dimensions))

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
                self._update(space.agents, function,
                            local_position, velocity)

                # Checking if agents meets the bounds limits
                space.clip_limits()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, local_position, hook=pre_evaluation)

                # Reducing exploration parameter
                self.c *= self.decay

                # Every iteration, we need to dump agents, local positions and best agent
                history.dump(agents=space.agents, local=local_position,
                            best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.file(f'Fitness: {space.best_agent.fit}')
                logger.file(f'Position: {space.best_agent.position}')

        return history

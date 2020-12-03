"""Pigeon-Inspired Optimization.
"""

import numpy as np
from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class PIO(Optimizer):
    """A PIO class, inherited from Optimizer.

    This is the designed class to define PIO-related
    variables and methods.

    References:
        H. Duan and P. Qiao.
        Pigeon-inspired optimization:a new swarm intelligence optimizerfor air robot path planning.
        International Journal of IntelligentComputing and Cybernetics (2014).

    """

    def __init__(self, algorithm='PIO', hyperparams=None):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> PIO.')

        # Override its parent class with the receiving hyperparams
        super(PIO, self).__init__(algorithm)

        # Number of mapping iterations
        self.n_c1 = 150

        # Number of landmark iterations
        self.n_c2 = 200

        # Map and compass factor
        self.R = 0.2

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def n_c1(self):
        """int: Number of mapping iterations.

        """

        return self._n_c1

    @n_c1.setter
    def n_c1(self, n_c1):
        if not isinstance(n_c1, int):
            raise e.TypeError('`n_c1` should be an integer')
        if n_c1 <= 0:
            raise e.ValueError('`n_c1` should be > 0')

        self._n_c1 = n_c1

    @property
    def n_c2(self):
        """int: Number of landmark iterations.

        """

        return self._n_c2

    @n_c2.setter
    def n_c2(self, n_c2):
        if not isinstance(n_c2, int):
            raise e.TypeError('`n_c2` should be an integer')
        if n_c2 < self.n_c1:
            raise e.ValueError('`n_c1` should be > `n_c2')

        self._n_c2 = n_c2

    @property
    def R(self):
        """float: Map and compass factor.

        """

        return self._R

    @R.setter
    def R(self, R):
        if not isinstance(R, (float, int)):
            raise e.TypeError('`R` should be a float or integer')
        if R < 0:
            raise e.ValueError('`R` should be >= 0')

        self._R = R

    def _update_velocity(self, position, best_position, velocity, iteration):
        """Updates a particle velocity (eq. 5).

        Args:
            position (np.array): Agent's current position.
            best_position (np.array): Global best position.
            velocity (np.array): Agent's current velocity.
            iteration (int): Current iteration.

        Returns:
            A new velocity.

        """

        # Generating random number
        r1 = r.generate_uniform_random_number()

        # Calculates new velocity
        new_velocity = velocity * np.exp(-self.R * (iteration + 1)) + r1 * (best_position - position)

        return new_velocity

    def _update_position(self, position, velocity):
        """Updates a pigeon position (eq. 6).

        Args:
            position (np.array): Agent's current position.
            velocity (np.array): Agent's current velocity.

        Returns:
            A new position.

        """

        # Calculates new position
        new_position = position + velocity

        return new_position

    def _calculate_center(self, agents):
        """Calculates the center position (eq. 8).

        Args:
            agents (list): List of agents.

        Returns:
            The center position.

        """

        # Creates an array to hold the cummulative position
        total_pos = np.zeros((agents[0].n_variables, agents[0].n_dimensions))

        # Initializes total fitness as zero
        total_fit = 0.0

        # Iterates through all agents
        for agent in agents:
            # Accumulates the position
            total_pos += agent.position * agent.fit

            # Accumulates the fitness
            total_fit += agent.fit

        # Calculates the center position
        center = total_pos / (self.n_p * total_fit + c.EPSILON)

        return center

    def _update_center_position(self, position, center):
        """Updates a pigeon position based on the center (eq. 9).

        Args:
            position (np.array): Agent's current position.
            center (np.array): Center position.

        Returns:
            A new center-based position.

        """

        # Generating random number
        r1 = r.generate_uniform_random_number()

        # Calculates new position based on center
        new_position = position + r1 * (center - position)

        return new_position

    def _update(self, agents, best_agent, velocity, iteration):
        """Method that wraps velocity and position updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            velocity (np.array): Array of current velocities.
            iteration (int): Current iteration.

        """

        # Checks if current iteration is smaller than mapping operator
        if iteration < self.n_c1:
            # Iterates through all agents
            for i, agent in enumerate(agents):
                # Updates current agent velocity
                velocity[i] = self._update_velocity(agent.position, best_agent.position, velocity[i], iteration)

                # Updates current agent position
                agent.position = self._update_position(agent.position, velocity[i])

        # Checks if current iteration is smaller than landmark operator
        elif iteration < self.n_c2:
            # Calculates the number of possible pigeons (eq. 7)
            self.n_p = int(self.n_p / 2) + 1

            # Sorts agents according to their fitness
            agents.sort(key=lambda x: x.fit)

            # Calculates the center position
            center = self._calculate_center(agents[:self.n_p])

            # Iterates through all agents
            for agent in agents:
                # Updates current agent position
                agent.position = self._update_center_position( agent.position, center)

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

        # Instantiating number of total pigeons
        self.n_p = space.n_agents

        # Instanciating array velocities
        velocity = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

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
                self._update(space.agents, space.best_agent, velocity, t)

                # Checking if agents meet the bounds limits
                space.clip_limits()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, hook=pre_evaluation)

                # Every iteration, we need to dump agents, local positions and best agent
                history.dump(agents=space.agents, best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.file(f'Fitness: {space.best_agent.fit}')
                logger.file(f'Position: {space.best_agent.position}')

        return history

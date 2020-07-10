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


class BPSO(Optimizer):
    """A BPSO class, inherited from Optimizer.

    This is the designed class to define boolean PSO-related
    variables and methods.

    References:
        F. Afshinmanesh, A. Marandi and A. Rahimi-Kian.
        A Novel Binary Particle Swarm Optimization Method Using Artificial Immune System.
        IEEE International Conference on Smart Technologies (2005). 

    """

    def __init__(self, algorithm='BPSO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> BPSO.')

        # Override its parent class with the receiving hyperparams
        super(BPSO, self).__init__(algorithm=algorithm)

        # Cognitive constant
        self.c1 = np.array([1])

        # Social constant
        self.c2 = np.array([1])

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def c1(self):
        """float: Cognitive constant.

        """

        return self._c1

    @c1.setter
    def c1(self, c1):
        if not isinstance(c1, np.ndarray):
            raise e.TypeError('`c1` should be a numpy array')

        self._c1 = c1

    @property
    def c2(self):
        """float: Social constant.

        """

        return self._c2

    @c2.setter
    def c2(self, c2):
        if not isinstance(c2, np.ndarray):
            raise e.TypeError('`c2` should be a numpy array')

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
            if 'c1' in hyperparams:
                self.c1 = hyperparams['c1']
            if 'c2' in hyperparams:
                self.c2 = hyperparams['c2']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: c1 = {self.c1}, c2 = {self.c2} | '
            f'Built: {self.built}.')

    def _update_velocity(self, position, best_position, local_position):
        """Updates a particle velocity.

        Args:
            position (np.array): Agent's current position.
            best_position (np.array): Global best position.
            local_position (np.array): Agent's local best position.

        Returns:
            A new velocity based on boolean bPSO's paper velocity update equation.

        """

        # Defining a random binary number
        r1 = r.generate_binary_random_number(position.shape)

        # Defining another random binary number
        r2 = r.generate_binary_random_number(position.shape)

        # Calculating the local partial
        local_partial = np.logical_and(self.c1, np.logical_xor(r1, np.logical_xor(local_position, position)))

        # Calculating the global partial
        global_partial = np.logical_and(self.c2, np.logical_xor(r2, np.logical_xor(best_position, position)))

        # Updating new velocity
        new_velocity = np.logical_or(local_partial, global_partial)

        return new_velocity

    def _update_position(self, position, velocity):
        """Updates a particle position.

        Args:
            position (np.array): Agent's current position.
            velocity (np.array): Agent's current velocity.

        Returns:
            A new position based on boolean bPSO's paper position update equation.

        """

        # Calculates new position
        new_position = np.logical_xor(position, velocity)

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
            velocity[i] = self._update_velocity(agent.position, best_agent.position, local_position[i])

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
            (space.n_agents, space.n_variables, space.n_dimensions), dtype=bool)

        # And also an array of velocities
        velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions), dtype=bool)

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
                            local_position, velocity)

                # Checking if agents meets the bounds limits
                space.clip_limits()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, local_position, hook=pre_evaluation)

                # Every iteration, we need to dump agents, local positions and best agent
                history.dump(agents=space.agents, local=local_position,
                             best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.file(f'Fitness: {space.best_agent.fit}')
                logger.file(f'Position: {space.best_agent.position}')

        return history

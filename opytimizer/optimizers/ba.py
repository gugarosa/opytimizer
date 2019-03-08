import copy

import numpy as np
import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class BA(Optimizer):
    """A BA class, inherited from Optimizer.

    This will be the designed class to define BA-related
    variables and methods.

    References:
        X.-S. Yang. A new metaheuristic bat-inspired algorithm. Nature inspired cooperative strategies for optimization (2010).
        
    Attributes:
        f_min (float): Minimum frequency range.
        f_max (float): Maximum frequency range.
        A (float): Loudness parameter.
        r (float): Social rate.
        frequency (np.array): An array holding particles' frequencies.
        velocity (np.array): An array holding particles' velocities.
        loudness (np.array): An array holding particles' loudnesses.
        pulse_rate (np.array): An array holding particles' pulse rates.

    Methods:
        _build(hyperparams): Sets an external function point to a class attribute.
        _update_frequency(min, max): Updates a single particle frequency (over a single variable).
        _update_velocity(agent_position, best_position, frequency, current_velocity): Updates a single particle
            velocity (over a single variable).
        _update_position(agent_position, current_velocity): Updates a single particle
            position (over a single variable).
        _update(self, agents, best_agent, function, iteration, frequency,
            velocity, loudness, pulse_rate): Updates the agents according to bat algorithm.
        run(space, function): Runs the optimization pipeline.

    """

    def __init__(self, hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): An hyperparams dictionary containing key-value
                parameters to meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> BA.')

        # Override its parent class with the receiving hyperparams
        super(BA, self).__init__(algorithm='BA')

        # Minimum frequency range
        self._f_min = 0

        # Maximum frequency range
        self._f_max = 2

        # Loudness parameter
        self._A = 0.5

        # Pulse rate
        self._r = 0.5

        # Particles' frequencies
        self._frequency = None

        # Particles' velocities
        self._velocity = None

        # Particles' loudnesses
        self._loudness = None

        # Particles' pulse rates
        self._pulse_rate = None

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def f_min(self):
        """Minimum frequency range.

        """

        return self._f_min

    @f_min.setter
    def f_min(self, f_min):
        self._f_min = f_min

    @property
    def f_max(self):
        """Maximum frequency range.

        """

        return self._f_max

    @f_max.setter
    def f_max(self, f_max):
        self._f_max = f_max

    @property
    def A(self):
        """Loudness parameter.

        """

        return self._A

    @A.setter
    def A(self, A):
        self._A = A

    @property
    def r(self):
        """Pulse rate.

        """

        return self._r

    @r.setter
    def r(self, r):
        self._r = r

    @property
    def frequency(self):
        """Particles' current frequencies.

        """

        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        self._frequency = frequency

    @property
    def velocity(self):
        """Particles' current velocities.

        """

        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = velocity

    @property
    def loudness(self):
        """Particles' current loudnesses.

        """

        return self._loudness

    @loudness.setter
    def loudness(self, loudness):
        self._loudness = loudness

    @property
    def pulse_rate(self):
        """Particles' current pulse rates.

        """

        return self._pulse_rate

    @pulse_rate.setter
    def pulse_rate(self, pulse_rate):
        self._pulse_rate = pulse_rate

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
            if 'f_min' in hyperparams:
                self.f_min = hyperparams['f_min']
            if 'f_max' in hyperparams:
                self.f_max = hyperparams['f_max']
            if 'A' in hyperparams:
                self.A = hyperparams['A']
            if 'r' in hyperparams:
                self.r = hyperparams['r']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: f_min = {self.f_min}, f_max = {self.f_max}, A = {self.A}, r = {self.r} | Built: {self.built}.')

    def _update_frequency(self, min_frequency, max_frequency):
        """Updates a single particle frequency (over a single variable).

        Args:
            min_frequency (float): Minimum frequency range.
            max_frequency (float): Maximum frequency range.

        Returns:
            A new frequency based on BA's paper equation 2.

        """

        # Generating beta random number
        beta = r.generate_uniform_random_number(0, 1)

        # Calculating new frequency
        # Note that we have to apply (min - max) instead of (max - min) or it will not converge
        new_frequency = min_frequency + (min_frequency - max_frequency) * beta

        return new_frequency

    def _update_velocity(self, agent_position, best_position, frequency, current_velocity):
        """Updates a single particle velocity (over a single variable).

        Args:
            agent_position (float): Agent's current position.
            best_position (float): Global best position.
            frequency (float): Agent's frequenct.
            current_velocity (float): Agent's current velocity.

        Returns:
            A new velocity based on on BA's paper equation 3.

        """

        # Calculates new velocity
        new_velocity = current_velocity + \
            (agent_position - best_position) * frequency

        return new_velocity

    def _update_position(self, agent_position, current_velocity):
        """Updates a single particle position (over a single variable).

        Args:
            agent_position (float): Agent's current position.
            current_velocity (float): Agent's current velocity.

        Returns:
            A new position based on BA's paper equation 4.

        """

        # Calculates new position
        new_position = agent_position + current_velocity

        return new_position

    def _update(self, agents, best_agent, function, iteration, frequency, velocity, loudness, pulse_rate):
        """Method that wraps velocity and position updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            iteration (int): Current iteration number.
            frequency (np.array): Array of frequencies.
            velocity (np.array): Array of current velocities.
            loudness (np.array): Array of loudnesses.
            pulse_rate (np.array): Array of pulse rates.

        """

        # Declaring alpha constant
        alpha = 0.9

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # Updating frequency
            frequency[i] = self._update_frequency(self.f_min, self.f_max)

            # Updating velocity
            velocity[i] = self._update_velocity(
                agent.position, best_agent.position, frequency[i], velocity[i])

            # Updating agent's position
            agent.position = self._update_position(agent.position, velocity[i])

            # Generating a random probability
            p = r.generate_uniform_random_number(0, 1, 1)

            # Generating a random number
            e = r.generate_uniform_random_number(-1, 1, 1)

            # Check if probability is bigger than current pulse rate
            if p > pulse_rate[i]:
                # Copying a temporary agent based on current agent
                temp_agent = copy.deepcopy(best_agent)

                # Generating new temporary agent's position
                # Based on BA's paper equation 5
                temp_agent.position = temp_agent.position + \
                    e * np.mean(loudness)
            else:
                # Copies a temporary agent from the best agent
                temp_agent = copy.deepcopy(agent)

                # Perform a random walk based on probability
                temp_agent.position = temp_agent.position + e

            # Evaluates temporary agent
            fit = function.pointer(temp_agent.position)

            # Checks if probability is smaller than loudness and if fit is better
            if p < loudness[i] and fit < agent.fit:
                # Copying the new solution to space's agent
                agent = copy.deepcopy(temp_agent)

                # Increasing pulse rate (Equation 6)
                pulse_rate[i] = self.r * (1 - np.exp(-alpha * iteration))

                # Decreasing loudness (Equation 6)
                loudness[i] = self.A * alpha

            # Checks if current agent fitness is the best in spce
            if agent.fit < best_agent.fit:
                # If yes, we have a new best agent
                best_agent = copy.deepcopy(agent)

    def run(self, space, function):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Instanciating array of frequencies
        self.frequency = r.generate_uniform_random_number(
            self.f_min, self.f_max, space.n_agents)

        # Instanciating array of velocities
        self.velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # And also an array of loudnesses
        self.loudness = r.generate_uniform_random_number(
            0, self.A, space.n_agents)

        # Finally, an array of pulse rates
        self.pulse_rate = r.generate_uniform_random_number(
            0, self.r, space.n_agents)

        # Initial search space evaluation
        self._evaluate(space, function)

        # We will define a History object for further dumping
        history = h.History()

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents, space.best_agent, function, t,
                         self.frequency, self.velocity, self.loudness, self.pulse_rate)

            # Checking if agents meets the bounds limits
            space.check_bound_limits(space.agents, space.lb, space.ub)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            # Every iteration, we need to dump the current space agents
            history.dump(space.agents)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

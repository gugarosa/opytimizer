import copy

import numpy as np
import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class CS(Optimizer):
    """A CS class, inherited from Optimizer.

    This will be the designed class to define CS-related
    variables and methods.

    References:
        
        
    Attributes:
        alpha (float): Step size.
        beta (float): Used to compute the Lévy distribution.
        p (float): Probability of replacing worst nests.

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

        logger.info('Overriding class: Optimizer -> CS.')

        # Override its parent class with the receiving hyperparams
        super(CS, self).__init__(algorithm='CS')

        # Step size
        self._alpha = 0.2

        # Lévy distribution parameter
        self._beta = 1.5

        # Switch probability
        self._p = 0.3

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def alpha(self):
        """Step size.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def beta(self):
        """Lévy distribution parameter.

        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta

    @property
    def p(self):
        """Switch probability.

        """

        return self._p

    @p.setter
    def p(self, p):
        self._p = p

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
            if 'alpha' in hyperparams:
                self.alpha = hyperparams['alpha']
            if 'beta' in hyperparams:
                self.beta = hyperparams['beta']
            if 'p' in hyperparams:
                self.p = hyperparams['p']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: alpha = {self.alpha}, beta = {self.beta}, p = {self.p} | Built: {self.built}.')

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

    def _calculate_nest_loss(self, size, probability):
        """
        """

        loss = round(size * (1 - probability))

        return loss

    def _update(self, agents, best_agent, function):
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

        # Generates an index for nest k
        k = int(r.generate_uniform_random_number(0, len(agents)-1))

        # Copies agent k to a temporary agent
        temp_agent = copy.deepcopy(agents[k])

        # Generates a Lévy distribution
        step = d.generate_levy_distribution(self.beta, agents[0].n_variables)

        # Expanding its dimension (this is a must as step is a vector)
        step = np.expand_dims(step, axis=1)

        # print(step.shape)
        # print(temp_agent.position.shape)

        # (Equation 1)
        temp_agent.position += self.alpha * step

        #
        temp_agent.fit = function.pointer(temp_agent.position)

        # Generates an index for nest l
        l = int(r.generate_uniform_random_number(0, len(agents)-1))

        #
        if (temp_agent.fit < agents[l].fit):
            #
            agents[l] = copy.deepcopy(temp_agent)

        #
        agents.sort(key = lambda x: x.fit, reverse=True)

        #
        loss = self._calculate_nest_loss(len(agents), self.p)

        for i in range(len(agents)-1, loss, -1):
            temp_agent = copy.deepcopy(best_agent)
            r1 = r.generate_uniform_random_number(0, 1)
            k = int(r.generate_uniform_random_number(0, len(agents)-1))
            l = int(r.generate_uniform_random_number(0, len(agents)-1))
            temp_agent.position += r1 * (agents[k].position - agents[l].position)

            temp_agent.fit = function.pointer(temp_agent.position)

            if (temp_agent.fit < agents[i].fit):
                agents[i] = copy.deepcopy(temp_agent)




    def run(self, space, function):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Initial search space evaluation
        self._evaluate(space, function)

        # We will define a History object for further dumping
        history = h.History()

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents, space.best_agent, function)

            # Checking if agents meets the bounds limits
            space.check_bound_limits(space.agents, space.lb, space.ub)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            # Every iteration, we need to dump the current space agents
            history.dump(space.agents)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

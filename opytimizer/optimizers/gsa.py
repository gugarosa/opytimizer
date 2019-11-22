import copy

import numpy as np

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class GSA(Optimizer):
    """A GSA class, inherited from Optimizer.

    This is the designed class to define GSA-related
    variables and methods.

    References:


    """

    def __init__(self, algorithm='GSA', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> GSA.')

        # Override its parent class with the receiving hyperparams
        super(GSA, self).__init__(algorithm=algorithm)

        # Initial gravity value
        self.G = 2.467

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def G(self):
        """float: Initial gravity.

        """

        return self._G

    @G.setter
    def G(self, G):
        if not (isinstance(G, float) or isinstance(G, int)):
            raise e.TypeError('`G` should be a float or integer')
        if G < 0:
            raise e.ValueError('`G` should be >= 0')

        self._G = G

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
            if 'G' in hyperparams:
                self.G = hyperparams['G']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: G = {self.G} | Built: {self.built}.')

    def _calculate_force(self, agents, mass, gravity):
        """
        """

        #
        force = [[gravity * (mass[i] * mass[j]) / (g.euclidean_distance(agents[i].position, agents[j].position) + c.EPSILON)
                  * (agents[j].position - agents[i].position) for j in range(len(agents))] for i in range(len(agents))]

        #
        force = np.asarray(force)

        #
        # force = np.sum(r.generate_uniform_random_number(size=(len(agents), agents[0].n_variables, agents[0].n_dimensions)) * force, axis=1)
        force = np.sum(r.generate_uniform_random_number() * force, axis=1)

        return force

    def _update_velocity(self, force, mass, velocity):
        """Updates an agent velocity.

        Args:
            position (float): Agent's current position.
            best_position (float): Global best position.
            frequency (float): Agent's frequenct.
            velocity (float): Agent's current velocity.

        Returns:
            A new velocity based on on GSA's paper equation 11.

        """

        #
        acceleration = force / (mass + c.EPSILON)

        #
        new_velocity = r.generate_uniform_random_number() * velocity + acceleration

        return new_velocity

    def _update_position(self, position, velocity):
        """Updates an agent position.

        Args:
            position (float): Agent's current position.
            velocity (float): Agent's current velocity.

        Returns:
            A new position based on GSA's paper equation 12.

        """

        # Calculates new position
        new_position = position + velocity

        return new_position

    def _update(self, agents, best_agent, function, velocity, iteration):
        """Method that wraps Bat Algorithm over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            velocity (np.array): Array of current velocities.
            iteration (int): Current iteration number.

        """

        # Sorting agents
        agents.sort(key=lambda x: x.fit)

        # Gathering the best and worst agents
        best, worst = agents[0].fit, agents[-1].fit

        # Calculating agents' masses
        mass = [(agent.fit - worst) / (best - worst) for agent in agents]

        # Normalizing agents' masses
        norm_mass = mass / np.sum(mass)

        # Calculating the current gravity
        gravity = self.G / (iteration + 1)

        # Calculating agents' attraction force
        force = self._calculate_force(agents, norm_mass, gravity)

        # Iterate through all agents
        for i, agent in enumerate(agents):
            # Updates current agent velocities
            velocity[i] = self._update_velocity(
                force[i], norm_mass[i], velocity[i])

            # Updates current agent positions
            agent.position = self._update_position(agent.position, velocity[i])

    def run(self, space, function, store_best_only=False, pre_evaluation_hook=None):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (boolean): If True, only the best agent of each iteration is stored in History.
            pre_evaluation_hook (function): A function that receives the optimizer, space and function
                and returns None. This function is executed before evaluating the function being optimized.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Creates an array of velocities
        velocity = np.zeros(
            (space.n_agents, space.n_variables, space.n_dimensions))

        # Check if there is a pre-evaluation hook
        if pre_evaluation_hook:
            # Applies the hook
            pre_evaluation_hook(self, space, function)

        # Initial search space evaluation
        self._evaluate(space, function)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents, space.best_agent, function, velocity, t)

            # Checking if agents meets the bounds limits
            space.check_limits()

            # Check if there is a pre-evaluation hook
            if pre_evaluation_hook:
                # Applies the hook
                pre_evaluation_hook(self, space, function)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            # Every iteration, we need to dump agents and best agent
            history.dump(agents=space.agents, best_agent=space.best_agent)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

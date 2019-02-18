import numpy as np
import copy

import opytimizer.math.random as r
import opytimizer.utils.common as c
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class PSO(Optimizer):
    """A PSO class, inherited from Optimizer.
    This will be the designed class to define PSO-related
    variables and methods.

    Properties:
        w (float): Inertia weight.
        c1 (float): First learning factor.
        c2 (float): Second learning factor.
        local_position (np.array): An array holding particle's local positions.
        velocity (np.array): An array holding particles' velocities.

    Methods:
        _build(hyperparams): Sets an external function point to a class
        attribute.
        _update_velocity():
        _update_position():

    """

    def __init__(self, hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): An hyperparams dictionary containing key-value
            parameters to meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> PSO.')

        # Override its parent class with the receiving hyperparams
        super(PSO, self).__init__(algorithm='PSO')

        # Inertia weight
        self._w = 0.7

        # First learning factor
        self._c1 = 1.7

        # Second learning factor
        self._c2 = 1.7

        # Particles' local positions
        self._local_position = None

        # Particles' velocities
        self._velocity = None

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def w(self):
        """Inertia weight.
        """

        return self._w

    @property
    def c1(self):
        """First learning factor.
        """

        return self._c1

    @property
    def c2(self):
        """Second learning factor.
        """

        return self._c2

    @property
    def local_position(self):
        """Particles' local positions.
        """

        return self._local_position

    @property
    def velocity(self):
        """Particles' velocities.
        """

        return self._velocity

    def _build(self, hyperparams):
        """This method will serve as the object building process.
        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            hyperparams (dict): An hyperparams dictionary containing key-value
            parameters to meta-heuristics.

        """

        logger.debug('Running private method: build()')

        # We need to save the hyperparams object for faster looking up
        self._hyperparams = hyperparams

        # If one can find any hyperparam inside its object,
        # set them as the ones that will be used
        if hyperparams:
            if 'w' in hyperparams:
                self._w = hyperparams['w']
            if 'c1' in hyperparams:
                self._c1 = hyperparams['c1']
            if 'c2' in hyperparams:
                self._c2 = hyperparams['c2']

        # Set built variable to 'True'
        self._built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self._algorithm} | Hyperparameters: w = {self._w}, c1 = {self._c1}, c2 = {self._c2} | Built: {self._built}')

    def _update_velocity(self, agent_position, best_position, local_position, current_velocity):
        """
        """

        # Generating first random number
        r1 = r.generate_uniform_random_number(0, 1)

        # Generating second random number
        r2 = r.generate_uniform_random_number(0, 1)

        new_velocity = self.w * current_velocity + self.c1 * r1 * (local_position - agent_position) + self.c2 * r2 * (best_position - agent_position)

        return new_velocity

    def _update_position(self, agent_position, current_velocity):
        """
        """

        new_position = agent_position + current_velocity

        return new_position

    def _update(self, agents, best_agent, local_position, velocity):
        """
        """

        for i, agent in enumerate(agents):
            for var in range(agent.n_variables):
                velocity[i][var] = self._update_velocity(agent._position[var], best_agent._position[var], local_position[i][var], velocity[i][var])
                agent._position[var] = self._update_position(agent._position[var], velocity[i][var])
                

    def _evaluate(self, space, function, local_position):
        """Evaluates the search space according to the objective function.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        """

        for i, agent in enumerate(space.agents):
            fit = function.pointer(agent._position)
            if fit < agent._fit:
                agent._fit = fit
                local_position[i] = copy.deepcopy(agent._position)
            if agent._fit < space._best_agent._fit:
                print('Yes')
                space._best_agent = copy.deepcopy(agent)
                space._best_agent._position = copy.deepcopy(local_position[i])

        # # We need to evaluate every agent
        # for agent, local_position in zip(space.agents, self.local_position):
        #     # We apply agent's values as the function's input
        #     fit = function.pointer(agent.position)

        #     # If current fitness is better than previous agent's fitness
        #     if (fit < agent.fit):
        #         agent.fit = fit
        #         # Still missing on updating local position
        #         local_position = copy.deepcopy(agent.position)

        #     # Finally, we can update current agent's fitness

        #     # If agent's fitness is the best among the space
        #     if (agent.fit < space.best_agent.fit):
        #         # We update space's best agent
        #         space.best_agent = agent

    def run(self, space, function):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        """

        # Instanciating array of local positions
        self._local_position = np.zeros((space.n_agents, space.n_variables, space.n_dimensions))

        # And also an array of velocities
        self._velocity = np.zeros((space.n_agents, space.n_variables))

        # Initial search space evaluation
        self._evaluate(space, function, self._local_position)

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1} out of {space.n_iterations}')

            self._update(space._agents, space._best_agent, self._local_position, self._velocity)

            c.check_bound_limits(space.agents, space.lb, space.ub)
            # Updating agents' position
            # self._update(space.agents, space.best_agent, self._local_position, self._velocity)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function, self._local_position)


            logger.info(f'Fitness: {space.best_agent.fit}')
            print(f'Position: {space.best_agent.position}')



import numpy as np
import opytimizer.utils.common as c
import opytimizer.utils.logging as l
import opytimizer.utils.random as r
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class PSO(Optimizer):
    """A PSO class, inherited from Optimizer.

    This will be the designed class to define PSO-related
    variables and methods.

    Properties:
        w (float): Inertia weight parameter.

    Methods:
        _build(hyperparams): Sets an external function point to a class
        attribute.

    """

    def __init__(self, hyperparams=None):
        """Initialization method.

        Args:
            hyperparams (dict): An hyperparams dictionary containing key-value
            parameters to meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> PSO')

        # Override its parent class with the receiving hyperparams
        super(PSO, self).__init__(algorithm='PSO')

        # Default algorithm hyperparameters
        self.w = 0.7
        self.c1 = 1.7
        self.c2 = 1.7

        self.local_position = None
        self.velocity = None

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    def _build(self, hyperparams):
        """This method will serve as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            hyperparams (dict): An hyperparams dictionary containing key-value
            parameters to meta-heuristics.

        """

        logger.debug('Running private method: build()')

        # We need to save the hyperparams object for faster
        # looking up
        self.hyperparams = hyperparams

        # If one can find any hyperparam inside its object,
        # set them as the ones that will be used
        if self.hyperparams:
            if 'w' in self.hyperparams:
                self.w = self.hyperparams['w']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(f'Algorithm: {self.algorithm} | Hyperparameters: w = {self.w} | Built: {self.built}')

    def __update_velocity(self, agent, best_agent, local_position, velocity):

        r1 = r.generate_uniform_random_number(0, 1)
        r2 = r.generate_uniform_random_number(0, 1)

        for v in range(agent.n_variables):
            velocity[v] = self.w * velocity[v] + self.c1 * r1 * (local_position[v] - agent.position[v]) + self.c2 * r2 * (best_agent.position[v] - agent.position[v])


    def __update_position(self, agent, velocity):
        """Updates the actual position of a agent's decision variable.

        Args:
            agent (Agent): Agent to be updated.
            var (int): Index of decision variable.
        """

        # One can find this equation on Kennedy & Eberhart PSO paper
        # Not the true one yet!
        for v in range(agent.n_variables):
            agent.position[v] = agent.position[v] + velocity[v]

    def _update(self, agents, best_agent):
        """Updates the agents' position array.

        Args:
            agents ([Agents]): A list of agents that will be updated.

        """

        # We need to update every agent
        for agent, local_position, velocity in zip(agents, self.local_position, self.velocity):
            # For PSO, we need to update its position
            self.__update_velocity(agent, best_agent, local_position, velocity)
            self.__update_position(agent, velocity)


    def _evaluate(self, space, function):
        """Evaluates the search space according to the objective function.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
        
        """


        # We need to evaluate every agent
        for agent, local_position in zip(space.agents, self.local_position):
            # We apply agent's values as the function's input
            fit = function.pointer(agent.position)

            # If current fitness is better than previous agent's fitness
            if (fit < agent.fit):
                agent.fit = fit
                # Still missing on updating local position
                for v in range(space.n_variables):
                    local_position[v] = agent.position[v]

            # Finally, we can update current agent's fitness
            

            # If agent's fitness is the best among the space
            if (agent.fit < space.best_agent.fit):
                # We update space's best agent
                space.best_agent = agent

    def run(self, space, function):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        """

        self.local_position = np.zeros((space.n_agents, space.n_variables))
        self.velocity = np.zeros((space.n_agents, space.n_variables))

        # Initial search space evaluation 
        self._evaluate(space, function)
        
        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1} out of {space.n_iterations}')

            # Updating agents' position
            self._update(space.agents, space.best_agent)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            c.check_bound_limits(space.agents, space.lb, space.ub)

            logger.info(f'Fitness: {space.best_agent.fit}')
            print(f'Position: {space.best_agent.position}')

        
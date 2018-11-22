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
        self.w = 2.0

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

    def __update_position(self, agent, var):
        """Updates the actual position of a agent's decision variable.

        Args:
            agent (Agent): Agent to be updated.
            var (int): Index of decision variable.
        """

        # One can find this equation on Kennedy & Eberhart PSO paper
        # Not the true one yet!
        agent.position[var] = agent.position[var] * r.generate_uniform_random_number(0, 1)

    def _update(self, agents):
        """Updates the agents' position array.

        Args:
            agents ([Agents]): A list of agents that will be updated.

        """

        # We need to update every agent
        for agent in agents:
            # And also every decision variable of this agent
            for var in range(agent.n_variables):
                # For PSO, we need to update its position
                self.__update_position(agent, var)
                # And its velocity


    def _evaluate(self, space, function):
        """Evaluates the search space according to the objective function.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
        
        """

        # We need to evaluate every agent
        for agent in space.agents:
            # We apply agent's values as the function's input
            fit = function.pointer(agent.position)

            # If current fitness is better than previous agent's fitness
            if (fit < agent.fit):
                # Still missing on updating local position
                pass

            # Finally, we can update current agent's fitness
            agent.fit = fit

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

        # Initial search space evaluation 
        self._evaluate(space, function)
        
        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1} out of {space.n_iterations}')

            # Updating agents' position
            self._update(space.agents)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            logger.info(f'Fitness: {space.best_agent.fit}')
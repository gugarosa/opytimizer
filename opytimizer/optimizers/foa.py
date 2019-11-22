import copy

import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.agent import Agent
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class FOA(Optimizer):
    """A FOA class, inherited from Optimizer.

    This is the designed class to define FOA-related
    variables and methods.

    References:
        

    """

    def __init__(self, algorithm='FOA', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> FOA.')

        # Override its parent class with the receiving hyperparams
        super(FOA, self).__init__(algorithm=algorithm)

        # Randomization parameter
        self.life_time = 5

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def life_time(self):
        """float: Randomization parameter.

        """

        return self._life_time

    @life_time.setter
    def life_time(self, life_time):
        if not isinstance(life_time, int):
            raise e.TypeError('`life_time` should be an integer')
        if life_time <= 0:
            raise e.ValueError('`life_time` should be > 0')

        self._life_time = life_time

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
            if 'life_time' in hyperparams:
                self.life_time = hyperparams['life_time']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: life_time = {self.life_time} | Built: {self.built}.')

    def _local_seeding(self, agent):
        """
        """

        a = copy.deepcopy(agent)

        return a


    def _limit_population(self, agents, ages, size):
        candidates = []
        candidates_ages = []

        #
        sorted_agents = sorted(zip(agents, ages), key=lambda x: x[0].fit)

        print(sorted_agents)

        for sorted_agent in sorted_agents:
            if sorted_agent[1] > self.life_time:
                candidates.append(sorted_agent[0])
                candidates_ages.append(sorted_agent[1])
            if len(candidates) == size:
                break
            
        return candidates, candidates_ages


    def _update(self, space, function, ages):
        """Method that wraps Forest Optimization Algorithm over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A function object.
            n_iterations (int): Maximum number of iterations.

        """

        #
        agents = copy.deepcopy(space.agents)

        for i, agent in enumerate(space.agents):
            if ages[i] == 0:
                for j in range(3):
                    agents.append(self._local_seeding(agent))
                    ages = np.append(ages, 0)
            ages[i] += 1

        candidates, candidates_ages = self._limit_population(agents, ages, space.n_agents)

        self._global_seeding()


        return ages



        #
        # space.agents = copy.deepcopy(agents)


    def run(self, space, function):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Instanciating array of ages counter
        ages = np.zeros(space.n_agents)

        # Initial search space evaluation
        self._evaluate(space, function)

        # We will define a History object for further dumping
        history = h.History()

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            ages = self._update(space, function, ages)

            # Checking if agents meets the bounds limits
            space.check_limits()

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            # Every iteration, we need to dump agents and best agent
            history.dump(agents=space.agents, best_agent=space.best_agent)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

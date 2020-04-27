import copy
import numpy as np

import opytimizer.math.distribution as d
import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class AEO(Optimizer):
    """An AEO class, inherited from Optimizer.

    This is the designed class to define AEO-related
    variables and methods.

    References:
        X.-S. Yang. Flower pollination algorithm for global optimization.
        International conference on unconventional computing and natural computation (2012).

    """

    def __init__(self, algorithm='AEO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Override its parent class with the receiving hyperparams
        super(AEO, self).__init__(algorithm)

        # Lévy flight control parameter
        self.beta = 1.5

        # Lévy flight scaling factor
        self.eta = 0.2

        # Probability of local pollination
        self.p = 0.8

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def beta(self):
        """float: Lévy flight control parameter.

        """

        return self._beta

    @beta.setter
    def beta(self, beta):
        if not (isinstance(beta, float) or isinstance(beta, int)):
            raise e.TypeError('`beta` should be a float or integer')
        if beta < 0:
            raise e.ValueError('`beta` should be >= 0')

        self._beta = beta

    @property
    def eta(self):
        """float: Lévy flight scaling factor.

        """

        return self._eta

    @eta.setter
    def eta(self, eta):
        if not (isinstance(eta, float) or isinstance(eta, int)):
            raise e.TypeError('`eta` should be a float or integer')
        if eta < 0:
            raise e.ValueError('`eta` should be >= 0')

        self._eta = eta

    @property
    def p(self):
        """float: Probability of local pollination.

        """

        return self._p

    @p.setter
    def p(self, p):
        if not (isinstance(p, float) or isinstance(p, int)):
            raise e.TypeError('`p` should be a float or integer')
        if p < 0 or p > 1:
            raise e.ValueError('`p` should be between 0 and 1')

        self._p = p

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
            if 'beta' in hyperparams:
                self.beta = hyperparams['beta']
            if 'eta' in hyperparams:
                self.eta = hyperparams['eta']
            if 'p' in hyperparams:
                self.p = hyperparams['p']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | '
            f'Hyperparameters: beta = {self.beta}, eta = {self.eta}, p = {self.p} | '
            f'Built: {self.built}.')

    def _production(self, agent, best_agent, iteration, n_iterations):
        """
        """

        #
        a = copy.deepcopy(agent)

        #
        alpha = (1 - iteration / n_iterations) * r.generate_uniform_random_number()

        # For every possible decision variable
        for j, (lb, ub) in enumerate(zip(a.lb, a.ub)):
            # Updates its position
            a.position[j] = (1 - alpha) * best_agent.position[j] + alpha * r.generate_uniform_random_number(lb, ub, a.n_dimensions)

        return a

    def _herbivore_consumption(self, agent, producer, C):
        """
        """

        #
        a = copy.deepcopy(agent)

        #
        a.position += C * (agent.position - producer.position)

        return a

    def _omnivore_consumption(self, agent, producer, consumer, C):
        """
        """
        
        #
        a = copy.deepcopy(agent)

        #
        r2 = r.generate_uniform_random_number()

        #
        a.position += C * r2 * (a.position - producer.position) + (1 - r2) * (a.position - consumer.position)

        return a

    def _carnivore_consumption(self, agent, consumer, C):

        #
        a = copy.deepcopy(agent)

        #
        a.position += C * (a.position - consumer.position) 

        return a


    def _update_composition(self, agents, best_agent, function, iteration, n_iterations):
        """Method that wraps production and consumption updates over all agents and variables.

        Args:
            agents (list): List of agents.
            best_agent (Agent): Global best agent.
            function (Function): A Function object that will be used as the objective function.

        """

        # Iterate through all agents
        for i, agent in enumerate(agents):
            #
            if i == 0:
                #
                a = self._production(agent, best_agent, iteration, n_iterations)
            
            #
            else:
                #
                r1 = r.generate_uniform_random_number()

                #
                v1 = r.generate_gaussian_random_number(0, 1)

                #
                v2 = r.generate_gaussian_random_number(0, 1)
                
                #
                C = 0.5 * v1 / np.abs(v2)

                #
                if r1 < 1/3:
                    #
                    a = self._herbivore_consumption(agent, agents[0], C)
                
                #
                elif 1/3 <= r1 <= 2/3:
                    #
                    j = int(r.generate_uniform_random_number(1, i))

                    #
                    a = self._omnivore_consumption(agent, agents[0], agents[j], C)
                
                #
                else:
                    #
                    j = int(r.generate_uniform_random_number(1, i))

                    #
                    a = self._carnivore_consumption(agent, agents[j], C)
            
            # Check agent limits
            a.clip_limits()

            # Calculates the fitness for the temporary position
            a.fit = function.pointer(a.position)

            # If new fitness is better than agent's fitness
            if a.fit < agent.fit:
                # Copy its position to the agent
                agent.position = copy.deepcopy(a.position)

                # And also copy its fitness
                agent.fit = copy.deepcopy(a.fit)  

    def _update_decomposition():
        """
        """

        pass      

    def run(self, space, function, store_best_only=False, pre_evaluation_hook=None):
        """Runs the optimization pipeline.

        Args:
            space (Space): A Space object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.
            store_best_only (bool): If True, only the best agent of each iteration is stored in History.
            pre_evaluation_hook (callable): This function is executed before evaluating the function being optimized.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

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

            # Updating agents within the composition step
            self._update_composition(space.agents, space.best_agent, function, t, space.n_iterations)

            # Checking if agents meets the bounds limits
            space.clip_limits()

            # Check if there is a pre-evaluation hook
            if pre_evaluation_hook:
                # Applies the hook
                pre_evaluation_hook(self, space, function)

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function)

            # # Updating agents within the decomposition step
            # self._update_decomposition(space.agents, space.best_agent, function)

            # # Checking if agents meets the bounds limits
            # space.clip_limits()

            # # Check if there is a pre-evaluation hook
            # if pre_evaluation_hook:
            #     # Applies the hook
            #     pre_evaluation_hook(self, space, function)

            # # After the update, we need to re-evaluate the search space
            # self._evaluate(space, function)

            # Every iteration, we need to dump agents and best agent
            history.dump(agents=space.agents, best_agent=space.best_agent)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

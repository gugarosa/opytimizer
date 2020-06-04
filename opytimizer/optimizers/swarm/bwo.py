import copy
import random

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core import agent
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class BWO(Optimizer):
    """A BWO class, inherited from Optimizer.

    This is the designed class to define BWO-related
    variables and methods.

    References:
        V. Hayyolalam and A. Kazem. 
        Black Widow Optimization Algorithm: A novel meta-heuristic approach for solving engineering optimization problems.
        Engineering Applications of Artificial Intelligence (2020). 

    """

    def __init__(self, algorithm='BWO', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> BWO.')

        # Override its parent class with the receiving hyperparams
        super(BWO, self).__init__(algorithm=algorithm)

        # Procreating rate
        self.pp = 0.6

        # Cannibalism rate
        self.cr = 0.44

        # Mutation rate
        self.pm = 0.4

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def pp(self):
        """float: Procreating rate.

        """

        return self._pp

    @pp.setter
    def pp(self, pp):
        if not (isinstance(pp, float) or isinstance(pp, int)):
            raise e.TypeError('`pp` should be a float or integer')
        if pp < 0 or pp > 1:
            raise e.ValueError('`pp` should be between 0 and 1')

        self._pp = pp

    @property
    def cr(self):
        """float: Cannibalism rate.

        """

        return self._cr

    @cr.setter
    def cr(self, cr):
        if not (isinstance(cr, float) or isinstance(cr, int)):
            raise e.TypeError('`cr` should be a float or integer')
        if cr < 0:
            raise e.ValueError('`cr` should be >= 0')

        self._cr = cr

    @property
    def pm(self):
        """float: Mutation rate.

        """

        return self._pm

    @pm.setter
    def pm(self, pm):
        if not (isinstance(pm, float) or isinstance(pm, int)):
            raise e.TypeError('`pm` should be a float or integer')
        if pm < 0 or pm > 1:
            raise e.ValueError('`pm` should be between 0 and 1')

        self._pm = pm

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
            if 'pp' in hyperparams:
                self.pp = hyperparams['pp']
            if 'cr' in hyperparams:
                self.cr = hyperparams['cr']
            if 'pm' in hyperparams:
                self.pm = hyperparams['pm']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | '
            f'Hyperparameters: pp = {self.pp}, cr = {self.cr}, pm = {self.pm} | '
            f'Built: {self.built}.')

    def _procreating(self, x1, x2):
        """Performs the reproduction between a pair of parents.

        Args:
            x1 (Agent): Father to produce the offsprings.
            x2 (Agent): Mother to produce the offsprings.

        Returns:
            Two generated offsprings based on parents.

        """

        # Makes a deep copy of father and mother
        y1, y2 = copy.deepcopy(x1), copy.deepcopy(x2)

        # Generates another uniform random number
        alpha = r.generate_uniform_random_number()

        # Calculates the crossover based on a linear combination between father and mother
        y1.position = alpha * x1.position + (1 - alpha) * x2.position

        # Calculates the crossover based on a linear combination between father and mother
        y2.position = alpha * x2.position + (1 - alpha) * x1.position

        return y1, y2

    def _mutation(self, alpha):
        """Performs the mutation over offsprings.

        Args:
            alpha (Agent): First offspring.

        Returns:
            The mutated offspring.

        """
        
        beta = copy.deepcopy(alpha)

        # TODO: select non duplicate random parents
        r1, r2 = random.sample(range(len(alpha.position)), 2)
        
        beta.position[r1], beta.position[r2] = beta.position[r2], beta.position[r1]
        
        return beta

    def _update(self, agents, n_variables, function):
        """Method that wraps procreation, cannibalism and mutation over all agents and variables.
        
        Args:
            agents (list): List of agents.
            n_variables (int): Number of decision variables.
            function (Function): A Function object that will be used as the objective function.

        """
        
        # Retrieving the number of agents
        n_agents = len(agents)
        
        # Number of reproduction
        nr = int(n_agents * self.pp)
        
        # Number of mutation children
        nm = int(n_agents * self.pm)
        
        # Sorting agents
        agents.sort(key=lambda x: x.fit)
        
        # Select the best nr solutions in pop and save them in pop1
        pop1 = copy.deepcopy(agents[:nr])
        pop2 = []
        
        for i in range(0, nr):

            father, mother = random.sample(pop1, 2)
            
            new_agents = []
            
            for j in range(0, int(n_variables/2)):
                
                # Performs the procreating
                alpha, beta = self._procreating(father, mother)
                
                # Calculates new fitness for `alpha`
                alpha.fit = function(alpha.position)

                # Calculates new fitness for `beta`
                beta.fit = function(beta.position)

                new_agents.extend([copy.deepcopy(mother), alpha, beta])

            # Sorting agents
            new_agents.sort(key=lambda x: x.fit)
            
            new_agents = new_agents[:int(self.cr * len(new_agents))]
            pop2.extend(new_agents)
        
        for i in range(0, nm):
            
            rand = int(r.generate_uniform_random_number(0, nr-1))
            
            # Performs the mutation
            agent = self._mutation(pop1[rand])
            
            # Calculates new fitness for `temp`
            agent.fit = function(agent.position)
            
            pop2.extend([agent])
            
        # Joins both populations
        agents += pop2

        # Sorting agents
        agents.sort(key=lambda x: x.fit)

        return agents[:n_agents]

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

        # Initial search space evaluation
        self._evaluate(space, function, hook=pre_evaluation)

        # We will define a History object for further dumping
        history = h.History(store_best_only)

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating agents
            self._update(space.agents, space.n_variables, function)

            # Checking if agents meets the bounds limits
            space.clip_limits()

            # After the update, we need to re-evaluate the search space
            self._evaluate(space, function, hook=pre_evaluation)

            # Every iteration, we need to dump agents, local positions and best agent
            history.dump(agents=space.agents, best_agent=space.best_agent)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

import copy

from tqdm import tqdm

import opytimizer.math.random as r
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
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
        if cr < 0 or cr > 1:
            raise e.ValueError('`cr` should be between 0 and 1')

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
        """Procreates a pair of parents into offsprings.

        Args:
            x1 (Agent): Father to produce the offsprings.
            x2 (Agent): Mother to produce the offsprings.

        Returns:
            Two generated offsprings based on parents.

        """

        # Makes a deep copy of father and mother
        y1, y2 = copy.deepcopy(x1), copy.deepcopy(x2)

        # Generates a uniform random number
        alpha = r.generate_uniform_random_number()

        # Calculates the first crossover
        y1.position = alpha * x1.position + (1 - alpha) * x2.position

        # Calculates the second crossover
        y2.position = alpha * x2.position + (1 - alpha) * x1.position

        return y1, y2

    def _mutation(self, alpha):
        """Performs the mutation over an offspring

        Args:
            alpha (Agent): Offspring to be mutated.

        Returns:
            The mutated offspring.

        """

        # Checks if the number of variables is bigger than one
        if alpha.n_variables > 1:
            # Samples a random integer
            r1 = r2 = int(r.generate_uniform_random_number(0, alpha.n_variables))

            # While the sampled intergers are equal
            while r1 == r2:
                # Continues to sample the second integer
                r2 = int(r.generate_uniform_random_number(0, alpha.n_variables))

            # Swaps the randomly selected variables
            alpha.position[r1], alpha.position[r2] = alpha.position[r2], alpha.position[r1]
        
        return alpha

    def _update(self, agents, n_variables, function):
        """Method that wraps procreation, cannibalism and mutation over all agents and variables.
        
        Args:
            agents (list): List of agents.
            n_variables (int): Number of decision variables.
            function (Function): A Function object that will be used as the objective function.

        """
        
        # Retrieving the number of agents
        n_agents = len(agents)
        
        # Calculates the number agents that reproduces, are cannibals and mutates
        n_reproduct, n_cannibals, n_mutate = int(n_agents * self.pp), int(n_agents * self.cr), int(n_agents * self.pm)
        
        # Sorting agents
        agents.sort(key=lambda x: x.fit)

        # Selecting the best solutions and saving in auxiliary population
        agents1 = copy.deepcopy(agents[:n_reproduct])
        
        # Creating an empty auxiliary population
        agents2 = []

        # For every possible reproducting agent
        for _ in range(0, n_reproduct):
            # Sampling random uniform integers as indexes
            idx = r.generate_uniform_random_number(0, n_agents, size=2)

            # Making a deepcopy of father and mother
            father, mother = copy.deepcopy(agents[int(idx[0])]), copy.deepcopy(agents[int(idx[1])])
            
            # Creating an empty list of auxiliary agents
            new_agents = []
            
            # For every possible pair of variables
            for _ in range(0, int(n_variables / 2)):
                # Procreates parents into two new offsprings
                y1, y2 = self._procreating(father, mother)

                # Checking `y1` limits
                y1.clip_limits()

                # Checking `y2` limits
                y2.clip_limits()
                
                # Calculates new fitness for `y1`
                y1.fit = function(y1.position)

                # Calculates new fitness for `y2`
                y2.fit = function(y2.position)

                # Appends the mother and mutated agents to the new population
                new_agents.extend([mother, y1, y2])

            # Sorting new population
            new_agents.sort(key=lambda x: x.fit)
            
            # Extending auxiliary population with the number of cannibals
            agents2.extend(new_agents[:n_cannibals])
        
        # For every possible mutating agent
        for _ in range(0, n_mutate):
            # Sampling a random uniform integer as index
            idx = int(r.generate_uniform_random_number(0, n_reproduct))
            
            # Performs the mutation
            alpha = self._mutation(agents1[idx])

            # Checking `alpha` limits
            alpha.clip_limits()
            
            # Calculates new fitness for `alpha`
            alpha.fit = function(alpha.position)
            
            # Appends the mutated agent to the auxiliary population
            agents2.extend([alpha])
            
        # Joins both populations
        agents += agents2

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

        # Initializing a progress bar
        with tqdm(total=space.n_iterations) as b:
            # These are the number of iterations to converge
            for t in range(space.n_iterations):
                logger.file(f'Iteration {t+1}/{space.n_iterations}')

                # Updating agents
                space.agents = self._update(space.agents, space.n_variables, function)

                # Checking if agents meets the bounds limits
                space.clip_limits()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, hook=pre_evaluation)

                # Every iteration, we need to dump agents, local positions and best agent
                history.dump(agents=space.agents, best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.file(f'Fitness: {space.best_agent.fit}')
                logger.file(f'Position: {space.best_agent.position}')

        return history

import copy

import numpy as np
from tqdm import tqdm

import opytimizer.math.distribution as d
import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class GA(Optimizer):
    """An GA class, inherited from Optimizer.

    This is the designed class to define GA-related
    variables and methods.

    References:
        M. Mitchell. An introduction to genetic algorithms. MIT Press (1998).

    """

    def __init__(self, algorithm='GA', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        # Override its parent class with the receiving hyperparams
        super(GA, self).__init__(algorithm)

        # Probability of selection
        self.p_selection = 0.75

        # Probability of mutation
        self.p_mutation = 0.25

        # Probability of crossover
        self.p_crossover = 0.5

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def p_selection(self):
        """float: Probability of selection.

        """

        return self._p_selection

    @p_selection.setter
    def p_selection(self, p_selection):
        if not (isinstance(p_selection, float) or isinstance(p_selection, int)):
            raise e.TypeError('`p_selection` should be a float or integer')
        if p_selection < 0 or p_selection > 1:
            raise e.ValueError('`p_selection` should be between 0 and 1')

        self._p_selection = p_selection

    @property
    def p_mutation(self):
        """float: Probability of mutation.

        """

        return self._p_mutation

    @p_mutation.setter
    def p_mutation(self, p_mutation):
        if not (isinstance(p_mutation, float) or isinstance(p_mutation, int)):
            raise e.TypeError('`p_mutation` should be a float or integer')
        if p_mutation < 0 or p_mutation > 1:
            raise e.ValueError('`p_mutation` should be between 0 and 1')

        self._p_mutation = p_mutation

    @property
    def p_crossover(self):
        """float: Probability of crossover.

        """

        return self._p_crossover

    @p_crossover.setter
    def p_crossover(self, p_crossover):
        if not (isinstance(p_crossover, float) or isinstance(p_crossover, int)):
            raise e.TypeError('`p_crossover` should be a float or integer')
        if p_crossover < 0 or p_crossover > 1:
            raise e.ValueError('`p_crossover` should be between 0 and 1')

        self._p_crossover = p_crossover

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
            if 'p_selection' in hyperparams:
                self.p_selection = hyperparams['p_selection']
            if 'p_mutation' in hyperparams:
                self.p_mutation = hyperparams['p_mutation']
            if 'p_crossover' in hyperparams:
                self.p_crossover = hyperparams['p_crossover']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | '
            f'Hyperparameters: p_selection = {self.p_selection}, p_mutation = {self.p_mutation}, '
            f'p_crossover = {self.p_crossover} | Built: {self.built}.')

    def _roulette_selection(self, n_agents, fitness):
        """Performs a roulette selection on the population.

        Args:
            n_agents (int): Number of agents allowed in the space.
            fitness (list): A fitness list of every agent.

        Returns:
            The selected indexes of the population

        """

        # Calculates the number of selected individuals
        n_individuals = int(n_agents * self.p_selection)

        # Checks if `n_individuals` is an odd number
        if n_individuals % 2 != 0:
            # If it is, increase it by one
            n_individuals += 1

        # Calculates the total fitness
        total_fitness = np.sum(fitness)

        # Calculates the probability of each fitness
        probs = [fit / total_fitness for fit in fitness]

        # Performs the selection process
        selected = d.generate_choice_distribution(
            n_agents, probs, n_individuals)

        return selected

    def _crossover(self, father, mother):
        """Performs the crossover between a pair of parents.

        Args:
            father (Agent): Father to produce the offsprings.
            mother (Agent): Mother to produce the offsprings.

        Returns:
            Two generated offsprings based on parents.

        """

        # Makes a deep copy of father and mother
        alpha, beta = copy.deepcopy(father), copy.deepcopy(mother)

        # Generates a uniform random number
        r1 = r.generate_uniform_random_number()

        # If random number is smaller than crossover probability
        if r1 < self.p_crossover:
            # Generates another uniform random number
            r2 = r.generate_uniform_random_number()

            # Calculates the crossover based on a linear combination between father and mother
            alpha.position = r2 * father.position + (1 - r2) * mother.position

            # Calculates the crossover based on a linear combination between father and mother
            beta.position = r2 * mother.position + (1 - r2) * father.position

        return alpha, beta

    def _mutation(self, alpha, beta):
        """Performs the mutation over offsprings.

        Args:
            alpha (Agent): First offspring.
            beta (Agent): Second offspring.

        Returns:
            Two mutated offsprings.

        """

        # For every decision variable
        for j in range(alpha.n_variables):
            # Generates a uniform random number
            r1 = r.generate_uniform_random_number()

            # If random number is smaller than probability of mutation
            if r1 < self.p_mutation:
                # Mutates the offspring
                alpha.position[j] *= r.generate_gaussian_random_number()

            # Generates another uniform random number
            r2 = r.generate_uniform_random_number()

            # If random number is smaller than probability of mutation
            if r2 < self.p_mutation:
                # Mutates the offspring
                beta.position[j] *= r.generate_gaussian_random_number()

        return alpha, beta

    def _update(self, agents, function):
        """Method that wraps selection, crossover and mutation over all agents and variables.

        Args:
            agents (list): List of agents.
            function (Function): A Function object that will be used as the objective function.

        """

        # Creating a list to hold the new population
        new_agents = []

        # Retrieving the number of agents
        n_agents = len(agents)

        # Calculates a list of fitness from every agent
        fitness = [agent.fit + c.EPSILON for agent in agents]

        # Selects the parents
        selected = self._roulette_selection(n_agents, fitness)

        # For every pair of selected parents
        for s in g.pairwise(selected):
            # Performs the crossover
            alpha, beta = self._crossover(agents[s[0]], agents[s[1]])

            # Performs the mutation
            alpha, beta = self._mutation(alpha, beta)

            # Checking `alpha` limits
            alpha.clip_limits()

            # Checking `beta` limits
            beta.clip_limits()

            # Calculates new fitness for `alpha`
            alpha.fit = function(alpha.position)

            # Calculates new fitness for `beta`
            beta.fit = function(beta.position)

            # Appends the mutated agents to the children
            new_agents.extend([alpha, beta])

        # Joins both populations
        agents += new_agents

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
                space.agents = self._update(space.agents, function)

                # Checking if agents meets the bounds limits
                space.clip_limits()

                # After the update, we need to re-evaluate the search space
                self._evaluate(space, function, hook=pre_evaluation)

                # Every iteration, we need to dump agents and best agent
                history.dump(agents=space.agents, best_agent=space.best_agent)

                # Updates the `tqdm` status
                b.set_postfix(fitness=space.best_agent.fit)
                b.update()

                logger.file(f'Fitness: {space.best_agent.fit}')
                logger.file(f'Position: {space.best_agent.position}')

        return history

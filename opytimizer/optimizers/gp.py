import copy

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.exception as e
import opytimizer.utils.history as h
import opytimizer.utils.logging as l
from opytimizer.core.optimizer import Optimizer

logger = l.get_logger(__name__)


class GP(Optimizer):
    """A GP class, inherited from Optimizer.

    This is the designed class to define GP-related
    variables and methods.

    References:
        J. Koza. Genetic programming: On the programming of computers by means of natural selection (1992).

    """

    def __init__(self, algorithm='GP', hyperparams={}):
        """Initialization method.

        Args:
            algorithm (str): Indicates the algorithm name.
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.

        """

        logger.info('Overriding class: Optimizer -> GP.')

        # Override its parent class with the receiving hyperparams
        super(GP, self).__init__(algorithm=algorithm)

        # Probability of reproduction
        self.reproduction = 0.3

        # Probability of mutation
        self.mutation = 0.4

        # Probability of crossover
        self.crossover = 0.4

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def reproduction(self):
        """float: Probability of reproduction.

        """

        return self._reproduction

    @reproduction.setter
    def reproduction(self, reproduction):
        self._reproduction = reproduction

    @property
    def mutation(self):
        """float: Probability of mutation.

        """

        return self._mutation

    @mutation.setter
    def mutation(self, mutation):
        self._mutation = mutation

    @property
    def crossover(self):
        """float: Probability of crossover.

        """

        return self._crossover

    @crossover.setter
    def crossover(self, crossover):
        self._crossover = crossover

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
            if 'reproduction' in hyperparams:
                self.reproduction = hyperparams['reproduction']
            if 'mutation' in hyperparams:
                self.mutation = hyperparams['mutation']
            if 'crossover' in hyperparams:
                self.crossover = hyperparams['crossover']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: reproduction = {self.reproduction}, mutation = {self.mutation}, crossover = {self.crossover} | Built: {self.built}.')

    def _selection(self, fits, k):
        """
        """

        #
        selected = []

        #
        for i in range(k):
            #
            possible = np.random.choice(fits, k)

            #
            selected.append(np.where(min(possible) == fits)[0][0])

        return selected

    
    def _reproduct(self, space, tmp_trees):
        """
        """

        #
        n_reproduction = int(space.n_trees * self.reproduction)

        #
        selected = self._selection(space.fit_trees, n_reproduction)

        #
        for (i, s) in enumerate(selected):
            #
            space.trees[i] = copy.deepcopy(tmp_trees[s])

    def mute(self, space, tree):

        p = 0.5

        m_tree = copy.deepcopy(tree)

        random = r.generate_uniform_random_number()

        point = int(r.generate_uniform_random_number(2, space.get_depth(tree)))

        c = 0
        flag = 0

        if p > random:
            new_tree = space.prefix(m_tree, point, flag, 'FUNCTION', c)
        else:
            new_tree = space.prefix(m_tree, point, flag, 'TERMINAL', c)

        if new_tree:
            tmp = space.grow(space.min_depth, space.max_depth)

            if flag:
                tmp2 = new_tree.left
            else:
                tmp2 = new_tree.right

            if flag:
                new_tree.left = tmp
                tmp.flag = 1
            else:
                new_tree.right = tmp
                tmp.flag = 0
            tmp.parent = new_tree
        else:
            m_tree = space.grow(space.min_depth, space.max_depth)

        return m_tree


    def _mutate(self, space, tmp_trees):
        """
        """

        #
        n_reproduction = int(space.n_trees * self.reproduction)

        #
        n_mutation = int(space.n_trees * self.mutation)

        #
        selected = self._selection(space.fit_trees, n_mutation)

        for (i, m) in enumerate(selected):
            
            if space.get_depth(tmp_trees[m]) > 1:
                space.trees[i+n_reproduction] = self.mute(space, tmp_trees[m])
            else:
                space.trees[i+n_reproduction] = space.grow(space.min_depth, space.max_depth)

    def _cross(self, space, tmp_trees):
        """
        """

        #
        n_crossover = int(space.n_trees * self.crossover)

        #
        selected = self._selection(space.fit_trees, n_crossover)
    
    def _update(self, space):
        """
        """

        #
        tmp_trees = copy.deepcopy(space.trees)

        #
        self._reproduct(space, tmp_trees)

        #
        self._mutate(space, tmp_trees)

        #
        self._cross(space, tmp_trees)
        

    def _evaluate(self, space, function):
        """Evaluates the search space according to the objective function.

        Args:
            space (TreeSpace): A TreeSpace object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        """

        # Creates a new temporary agent
        a = copy.deepcopy(space.agents[0])

        # Iterate through all trees
        for i, tree in enumerate(space.trees):
            # Runs through the tree and return a position array
            a.position = space.output(tree)

            # Checks the agent limits
            a.check_limits()

            # Calculate the fitness value of the temporary agent
            fit = function.pointer(a.position)

            # If fitness is better than tree's best fit
            if fit < space.fit_trees[i]:
                # Updates its current fitness to the newer one
                space.fit_trees[i] = fit

            # If tree's fitness is better than global fitness
            if space.fit_trees[i] < space.best_agent.fit:
                # Makes a deep copy of current tree's index to the space's best index
                space.best_index = i

                # Makes a deep copy of agent's best position to the best agent
                space.best_agent.position = copy.deepcopy(a.position)

                # Makes a deep copy of current tree fitness to the best agent
                space.best_agent.fit = copy.deepcopy(space.fit_trees[i])
    
    def run(self, space, function):
        """Runs the optimization pipeline.

        Args:
            space (TreeSpace): A TreeSpace object that will be evaluated.
            function (Function): A Function object that will be used as the objective function.

        Returns:
            A History object holding all agents' positions and fitness achieved during the task.

        """

        # Initial tree space evaluation
        self._evaluate(space, function)

        # We will define a History object for further dumping
        history = h.History()

        # These are the number of iterations to converge
        for t in range(space.n_iterations):
            logger.info(f'Iteration {t+1}/{space.n_iterations}')

            # Updating trees
            self._update(space)

            # After the update, we need to re-evaluate the tree space
            self._evaluate(space, function)

            # Every iteration, we need to dump the current space agents
            history.dump(space.agents, space.best_agent, space.best_index)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

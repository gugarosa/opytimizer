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

    
    def _reproduct(self, space, trees):
        """Reproducts a number of individuals through a tournament selection procedure.

        Args:
            space (TreeSpace): A TreeSpace object.
            trees (list): Temporary trees.

        """

        # Calculates a list of current trees' fitness
        fitness = [agent.fit for agent in space.agents]

        # Number of individuals to be reproducted
        n_individuals = int(space.n_trees * self.reproduction)

        # Gathers a list of selected individuals to be replaced
        selected = r.tournament_selection(fitness, n_individuals)

        # For every selected individual
        for s in selected:
            # Gathers the worst individual index
            worst = np.argmax(fitness)

            # Replace the individual by performing a deep copy on selected tree
            space.trees[worst] = copy.deepcopy(trees[s])

            # Replaces the worst individua fitness with a minimum value
            fitness[worst] = 0

    def mute(self, space, tree):

        p = 0.5

        m_tree = copy.deepcopy(tree)

        random = r.generate_uniform_random_number()

        point = int(r.generate_uniform_random_number(2, tree.n_nodes))

        c = 0
        flag = 0

        if p > random:
            new_tree = m_tree.prefix(m_tree, point, flag, 'FUNCTION', c)
        else:
            new_tree = m_tree.prefix(m_tree, point, flag, 'TERMINAL', c)

        if new_tree:
            tmp = space.grow(space.min_depth, space.max_depth)

            if flag:
                tmp2 = new_tree.left
            else:
                tmp2 = new_tree.right

            if flag:
                new_tree.left = tmp
                tmp.flag = True
            else:
                new_tree.right = tmp
                tmp.flag = False
            tmp.parent = new_tree
        else:
            m_tree = space.grow(space.min_depth, space.max_depth)

        return m_tree


    def _mutate(self, space, tmp_trees):
        """
        """

        fits = [agent.fit for agent in space.agents]

        #
        n_reproduction = int(space.n_trees * self.reproduction)

        #
        n_mutation = int(space.n_trees * self.mutation)

        #
        selected = r.tournament_selection(fits, n_mutation)

        for (i, m) in enumerate(selected):
            
            if tmp_trees[m].n_nodes > 1:
                space.trees[i+n_reproduction] = self.mute(space, tmp_trees[m])
            else:
                space.trees[i+n_reproduction] = space.grow(space.min_depth, space.max_depth)

    
    def _update(self, space):
        """Method that wraps reproduction, crossover and mutation operators over all trees.

        Args:
            space (TreeSpace): A TreeSpace object.

        """

        # Copying current trees to initiate a new generation
        new_trees = copy.deepcopy(space.trees)

        # Performs the reproduction
        self._reproduct(space, new_trees)

        # Performs the mutation
        self._mutate(space, new_trees)

        # Performs the crossover
        # self._cross(space, tmp_trees)
        

    def _evaluate(self, space, function):
        """Evaluates the search space according to the objective function.

        Args:
            space (TreeSpace): A TreeSpace object.
            function (Function): A Function object that will be used as the objective function.

        """

        # Iterate through all (trees, agents)
        for i, (tree, agent) in enumerate(zip(space.trees, space.agents)):
            # Runs through the tree and returns a position array
            agent.position = copy.deepcopy(tree.position)

            # Checks the agent limits
            agent.check_limits()

            # Calculates the fitness value of the agent
            agent.fit = function.pointer(agent.position)

            # If tree's fitness is better than global fitness
            if agent.fit < space.best_agent.fit:
                # Makes a deep copy of current tree
                space.best_tree = copy.deepcopy(tree)

                # Makes a deep copy of agent's position to the best agent
                space.best_agent.position = copy.deepcopy(agent.position)

                # Also, copies its fitness from agent's fitness
                space.best_agent.fit = copy.deepcopy(agent.fit)
    
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

            # Updating trees with designed operators
            self._update(space)

            # After the update, we need to re-evaluate the tree space
            self._evaluate(space, function)

            # Every iteration, we need to dump agents and best agent
            history.dump(agents=space.agents, best_agent=space.best_agent, best_tree=space.best_tree)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

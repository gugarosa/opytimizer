import copy

import numpy as np

import opytimizer.math.common as c
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
        self.p_reproduction = 0.3

        # Probability of mutation
        self.p_mutation = 0.4

        # Probability of crossover
        self.p_crossover = 0.4

        # Now, we need to build this class up
        self._build(hyperparams)

        logger.info('Class overrided.')

    @property
    def p_reproduction(self):
        """float: Probability of reproduction.

        """

        return self._p_reproduction

    @p_reproduction.setter
    def p_reproduction(self, p_reproduction):
        if not (isinstance(p_reproduction, float) or isinstance(p_reproduction, int)):
            raise e.TypeError('`p_reproduction` should be a float or integer')
        if p_reproduction < 0 or p_reproduction > 1:
            raise e.ValueError('`p_reproduction` should be between 0 and 1')
        self._p_reproduction = p_reproduction

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
            if 'p_reproduction' in hyperparams:
                self.p_reproduction = hyperparams['p_reproduction']
            if 'p_mutation' in hyperparams:
                self.p_mutation = hyperparams['p_mutation']
            if 'p_crossover' in hyperparams:
                self.p_crossover = hyperparams['p_crossover']

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Algorithm: {self.algorithm} | Hyperparameters: p_reproduction = {self.p_reproduction}, p_mutation = {self.p_mutation}, p_crossover = {self.p_crossover} | Built: {self.built}.')

    def _reproduction(self, space, agents, trees):
        """Reproducts a number of individuals pre-selected through a tournament procedure.

        Args:
            space (TreeSpace): A TreeSpace object.
            agents (list): Current iteration agents.
            trees (list): Current iteration trees.

        """

        # Calculates a list of current trees' fitness
        fitness = [agent.fit for agent in agents]

        # Number of individuals to be reproducted
        n_individuals = int(space.n_trees * self.p_reproduction)

        # Gathers a list of selected individuals to be replaced
        selected = r.tournament_selection(fitness, n_individuals)

        # For every selected individual
        for s in selected:
            # Gathers the worst individual index
            worst = np.argmax(fitness)

            # Replace the individual by performing a deep copy on selected tree and agent
            space.trees[worst] = trees[s]
            space.agents[worst] = copy.deepcopy(agents[s])

            # Replaces the worst individua fitness with a minimum value
            fitness[worst] = 0

    def _mutation(self, space, agents, trees):
        """Mutates a number of individuals pre-selected through a tournament procedure.

        Args:
            space (TreeSpace): A TreeSpace object.
            agents (list): Current iteration agents.
            trees (list): Current iteration trees.

        """

        # Calculates a list of current trees' fitness
        fitness = [agent.fit for agent in agents]

        # Number of individuals to be reproducted
        n_individuals = int(space.n_trees * self.p_mutation)

        # Gathers a list of selected individuals to be replaced
        selected = r.tournament_selection(fitness, n_individuals)

        # For every selected individual
        for s in selected:
            # Checks if the tree has more than one node
            if trees[s].n_nodes > 1:
                space.trees[s] = self._mutate(space, trees[s])

            # If there is only one node
            else:
                # Re-create it with a random tree
                space.trees[s] = space.grow(space.min_depth, space.max_depth)

    def _mutate(self, space, tree):
        """Actually performs the mutation on a single tree (Node).

        Args:
            space (TreeSpace): A TreeSpace object.
            trees (Node): A Node instance to be mutated.

        Returns:
            A mutated tree (Node).

        """

        # Calculating mutation point
        mutation_point = int(r.generate_uniform_random_number(2, tree.n_nodes))

        # Finds the node at desired mutation point
        sub_tree, flag = tree.find_node(mutation_point)

        # If the mutation point's parent is not a root (this may happen when the mutation point is a function),
        # and find_node() stops at a terminal node whose father is a root
        if sub_tree:
            # Creating a new random sub-tree
            branch = space.grow(space.min_depth, space.max_depth)

            # Checks if sub-tree should be positioned in the left
            if flag:
                # The left child will receive the sub-branch
                sub_tree.left = branch

                # And its flag will be True
                branch.flag = True

            # If `flag` is False
            else:
                # The right child will receive the sub-branch
                sub_tree.right = branch

                # And its flag will be False
                branch.flag = False

            # Connects the sub-branch to its parent
            branch.parent = sub_tree

        # Otherwise, if condition is false
        else:
            # The mutated tree will be a random tree
            tree = space.grow(space.min_depth, space.max_depth) 

        return tree

    def _crossover(self, space, agents, trees):
        """Crossover a number of individuals pre-selected through a tournament procedure.

        Args:
            space (TreeSpace): A TreeSpace object.
            agents (list): Current iteration agents.
            trees (list): Current iteration trees.

        """

        # Calculates a list of current trees' fitness
        fitness = [agent.fit for agent in agents]

        # Number of individuals to be reproducted
        n_individuals = int(space.n_trees * self.p_crossover)

        # Checks if `n_individuals is an odd number`
        if n_individuals % 2 != 0:
            # If it is, increase it by one
            n_individuals += 1

        # Gathers a list of selected individuals to be replaced
        selected = r.tournament_selection(fitness, n_individuals)

        # For every pair in selected individuals
        for s in c.pairwise(selected):
            # Checks if both trees have more than one node
            if (trees[s[0]].n_nodes > 1) and (trees[s[1]].n_nodes > 1):
                # Apply the crossover operation
                space.trees[s[0]], space.trees[s[1]] = self._cross(trees[s[0]], trees[s[1]])

    @profile
    def _cross(self, father, mother):
        """Actually performs the crossover over a father and mother nodes.

        Args:
            father (Node): A father's node to be crossed.
            mother (Node): A mother's node to be crossed.

        Returns:
            Two offsprings based on the crossover operator.

        """

        # Copying father tree to the father's offspring structure
        # father_offspring = copy.deepcopy(father)
        father_offspring = father

        # Calculating father's crossover point
        father_point = int(r.generate_uniform_random_number(2, father.n_nodes))

        # Finds the node at desired crossover point
        sub_father, flag_father = father_offspring.find_node(father_point)

        # Copying mother tree to the mother's offspring structure
        # mother_offspring = copy.deepcopy(mother)
        mother_offspring = mother

        # Calculating mother's crossover point
        mother_point = int(r.generate_uniform_random_number(2, mother.n_nodes))

        # Finds the node at desired crossover point
        sub_mother, flag_mother = mother_offspring.find_node(mother_point)

        # If there are crossover nodes
        if sub_father and sub_mother:
            # If father's node is positioned in the left
            if flag_father:
                # Gathers its left branch
                branch = sub_father.left

                # If mother's node is positioned in the left
                if flag_mother:
                    # Changes father's left node with mother's left node
                    sub_father.left = sub_mother.left

                    # And activates the flag
                    sub_mother.left.flag = True
                
                # If mother's node is positioned in the right
                else:
                    # Changes father's left node with mother's right node
                    sub_father.left = sub_mother.right

                    # Activates the flag
                    sub_mother.right.flag = True
            
            # If father's node is positioned in the right
            else:
                # Gathers its right branch
                branch = sub_father.right

                # If mother's node is positioned in the left
                if flag_mother:
                    # Changes father's right node with mother's left node
                    sub_father.right = sub_mother.left

                    # Deactivates the flag
                    sub_mother.left.flag = False

                # If mother's node is positioned in the right
                else:
                    # Changes father's right node with mother's right node
                    sub_father.right = sub_mother.right

                    # And deactivates the flag
                    sub_mother.right.flag = False

            # Finally, mother's parent will be the father's node
            sub_mother.parent = sub_father

            # Now, for creating the mother's offspring
            # Check if it is positioned in the left
            if flag_mother:
                # Applies the father's removed branch to mother's left child
                sub_mother.left = branch

                # Activates the flag
                branch.flag = True

            # If it is positioned in the right
            else:
                # Applies the father's removed branch to mother's right child
                sub_mother.right = branch

                # Deactivates the flag
                branch.flag = False
            
            # The branch's parent will be the mother's node
            branch.parent = sub_mother

        return father_offspring, mother_offspring

    def _update(self, space):
        """Method that wraps reproduction, crossover and mutation operators over all trees.

        Args:
            space (TreeSpace): A TreeSpace object.

        """

        # Copying current agents to initiate a new generation
        agents = copy.deepcopy(space.agents)

        # Copying current trees to initiate a new generation
        trees = copy.deepcopy(space.trees)

        # Performs the reproduction
        self._reproduction(space, agents, trees)

        # Performs the crossover
        # self._crossover(space, agents, trees)

        # Performs the mutation
        self._mutation(space, agents, trees)

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
            history.dump(agents=space.agents,
                         best_agent=space.best_agent, best_tree=space.best_tree)

            logger.info(f'Fitness: {space.best_agent.fit}')
            logger.info(f'Position: {space.best_agent.position}')

        return history

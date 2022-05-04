"""Genetic Programming.
"""

import copy
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

import opytimizer.math.general as g
import opytimizer.math.random as r
import opytimizer.utils.exception as e
from opytimizer.core import Optimizer
from opytimizer.core.function import Function
from opytimizer.core.node import Node
from opytimizer.core.space import Space
from opytimizer.spaces.tree import TreeSpace
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class GP(Optimizer):
    """A GP class, inherited from Optimizer.

    This is the designed class to define GP-related
    variables and methods.

    References:
        J. Koza. Genetic programming: On the programming of computers by means of natural selection (1992).

    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialization method.

        Args:
            params: Contains key-value parameters to the meta-heuristics.

        """

        logger.info("Overriding class: Optimizer -> GP.")

        # Overrides its parent class with the receiving params
        super(GP, self).__init__()

        # Probability of reproduction
        self.p_reproduction = 0.25

        # Probability of mutation
        self.p_mutation = 0.1

        # Probability of crossover
        self.p_crossover = 0.1

        # Nodes' prunning ratio
        self.prunning_ratio = 0

        # Builds the class
        self.build(params)

        logger.info("Class overrided.")

    @property
    def p_reproduction(self) -> float:
        """Probability of reproduction."""

        return self._p_reproduction

    @p_reproduction.setter
    def p_reproduction(self, p_reproduction: float) -> None:
        if not isinstance(p_reproduction, (float, int)):
            raise e.TypeError("`p_reproduction` should be a float or integer")
        if p_reproduction < 0 or p_reproduction > 1:
            raise e.ValueError("`p_reproduction` should be between 0 and 1")

        self._p_reproduction = p_reproduction

    @property
    def p_mutation(self) -> float:
        """Probability of mutation."""

        return self._p_mutation

    @p_mutation.setter
    def p_mutation(self, p_mutation: float) -> None:
        if not isinstance(p_mutation, (float, int)):
            raise e.TypeError("`p_mutation` should be a float or integer")
        if p_mutation < 0 or p_mutation > 1:
            raise e.ValueError("`p_mutation` should be between 0 and 1")

        self._p_mutation = p_mutation

    @property
    def p_crossover(self) -> float:
        """Probability of crossover."""

        return self._p_crossover

    @p_crossover.setter
    def p_crossover(self, p_crossover: float) -> None:
        if not isinstance(p_crossover, (float, int)):
            raise e.TypeError("`p_crossover` should be a float or integer")
        if p_crossover < 0 or p_crossover > 1:
            raise e.ValueError("`p_crossover` should be between 0 and 1")

        self._p_crossover = p_crossover

    @property
    def prunning_ratio(self) -> float:
        """Nodes' prunning ratio."""

        return self._prunning_ratio

    @prunning_ratio.setter
    def prunning_ratio(self, prunning_ratio: float) -> None:
        if not isinstance(prunning_ratio, (float, int)):
            raise e.TypeError("`prunning_ratio` should be a float or integer")
        if prunning_ratio < 0 or prunning_ratio > 1:
            raise e.ValueError("`prunning_ratio` should be between 0 and 1")

        self._prunning_ratio = prunning_ratio

    def _prune_nodes(self, n_nodes: int) -> int:
        """Prunes the amount of possible nodes used for mutation and crossover.

        Args:
            n_nodes: Number of current nodes.

        Returns:
            (int): Amount of prunned nodes.

        """

        # Calculates the amount of prunned nodes
        prunned_nodes = int(n_nodes * (1 - self.prunning_ratio))

        # If this amount is smaller than 2, it must be returned as 2
        if prunned_nodes <= 2:
            return 2

        return prunned_nodes

    def _reproduction(self, space: TreeSpace) -> None:
        """Reproducts a number of individuals pre-selected through a tournament procedure (p. 99).

        Args:
            space: A TreeSpace object.

        """

        # Calculates a list of current trees' fitness
        fitness = [agent.fit for agent in space.agents]

        # Number of individuals to be reproducted
        n_individuals = int(space.n_agents * self.p_reproduction)

        # Gathers a list of selected individuals to be replaced
        selected = g.tournament_selection(fitness, n_individuals)

        # For every selected individual
        for s in selected:
            # Gathers the worst individual index
            worst = np.argmax(fitness)

            # Replace the individual by performing a deep copy on selected tree
            space.trees[worst] = copy.deepcopy(space.trees[s])

            # We also need to copy the agent
            space.agents[worst] = copy.deepcopy(space.agents[s])

            # Replaces the worst individual fitness with a minimum value
            fitness[worst] = 0

    def _mutation(self, space: TreeSpace) -> None:
        """Mutates a number of individuals pre-selected through a tournament procedure.

        Args:
            space: A TreeSpace object.

        """

        # Calculates a list of current trees' fitness
        fitness = [agent.fit for agent in space.agents]

        # Number of individuals to be mutated
        n_individuals = int(space.n_agents * self.p_mutation)

        # Gathers a list of selected individuals to be replaced
        selected = g.tournament_selection(fitness, n_individuals)

        # For every selected individual
        for s in selected:
            # Gathers individual number of nodes
            n_nodes = space.trees[s].n_nodes

            # Checks if the tree has more than one node
            if n_nodes > 1:
                # Prunes the amount of maximum nodes
                max_nodes = self._prune_nodes(n_nodes)

                # Mutatets the individual
                space.trees[s] = self._mutate(space, space.trees[s], max_nodes)

            # If there is only one node
            else:
                # Re-create it with a random tree
                space.trees[s] = space.grow(space.min_depth, space.max_depth)

    def _mutate(self, space: TreeSpace, tree: Node, max_nodes: int) -> Node:
        """Actually performs the mutation on a single tree (p. 105).

        Args:
            space: A TreeSpace object.
            trees: A Node instance to be mutated.
            max_nodes: Maximum number of nodes to be searched.

        Returns:
            (Node): A mutated tree.

        """

        # Deep copying a new mutated tree from initial tree
        mutated_tree = copy.deepcopy(tree)

        # Calculates mutation point
        mutation_point = int(r.generate_uniform_random_number(2, max_nodes))

        # Finds the node at desired mutation point
        sub_tree, flag = mutated_tree.find_node(mutation_point)

        # If the mutation point's parent is not a root (this may happen when the mutation point is a function),
        # and find_node() stops at a terminal node whose father is a root
        if sub_tree:
            # Creates a new random sub-tree
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
            mutated_tree = space.grow(space.min_depth, space.max_depth)

        return mutated_tree

    def _crossover(self, space: TreeSpace) -> None:
        """Crossover a number of individuals pre-selected through a tournament procedure (p. 101).

        Args:
            space: A TreeSpace object.

        """

        # Calculates a list of current trees' fitness
        fitness = [agent.fit for agent in space.agents]

        # Number of individuals to be crossovered
        n_individuals = int(space.n_agents * self.p_crossover)

        # Checks if `n_individuals` is an odd number
        if n_individuals % 2 != 0:
            # If it is, increase it by one
            n_individuals += 1

        # Gathers a list of selected individuals to be replaced
        selected = g.tournament_selection(fitness, n_individuals)

        # For every pair in selected individuals
        for s in g.n_wise(selected):
            # Calculates the amount of father and mother nodes
            father_nodes = space.trees[s[0]].n_nodes
            mother_nodes = space.trees[s[1]].n_nodes

            # Checks if both trees have more than one node
            if (father_nodes > 1) and (mother_nodes > 1):
                # Prunning father and mother nodes
                max_f_nodes = self._prune_nodes(father_nodes)
                max_m_nodes = self._prune_nodes(mother_nodes)

                # Apply the crossover operation
                space.trees[s[0]], space.trees[s[1]] = self._cross(
                    space.trees[s[0]], space.trees[s[1]], max_f_nodes, max_m_nodes
                )

    def _cross(
        self, father: Node, mother: Node, max_father: int, max_mother: int
    ) -> Tuple[Node, Node]:
        """Actually performs the crossover over a father and mother nodes.

        Args:
            father: A father's node to be crossed.
            mother: A mother's node to be crossed.
            max_father: Maximum of nodes from father to be used.
            max_mother: Maximum of nodes from mother to be used.

        Returns:
            (Tuple[Node, Node]): Two offsprings based on the crossover operator.

        """

        # Copiesing father tree to the father's offspring structure
        father_offspring = copy.deepcopy(father)

        # Calculates father's crossover point
        father_point = int(r.generate_uniform_random_number(2, max_father))

        # Finds the node at desired crossover point
        sub_father, flag_father = father_offspring.find_node(father_point)

        # Copiesing mother tree to the mother's offspring structure
        mother_offspring = copy.deepcopy(mother)

        # Calculates mother's crossover point
        mother_point = int(r.generate_uniform_random_number(2, max_mother))

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

        return father, mother

    def evaluate(self, space: Space, function: Function) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A TreeSpace object.
            function: A Function object that will be used as the objective function.

        """

        # Iterates through all (trees, agents)
        for (tree, agent) in zip(space.trees, space.agents):
            # Runs through the tree and returns a position array
            agent.position = copy.deepcopy(tree.position)

            # Checks the agent limits
            agent.clip_by_bound()

            # Calculates the fitness value of the agent
            agent.fit = function(agent.position)

            # If tree's fitness is better than global fitness
            if agent.fit < space.best_agent.fit:
                # Makes a deep copy of current tree, position and fitness
                space.best_tree = copy.deepcopy(tree)
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: Space) -> None:
        """Wraps Genetic Programming over all trees and variables.

        Args:
            space: TreeSpace containing agents and update-related information.

        """

        # Performs the reproduction
        self._reproduction(space)

        # Performs the crossover
        self._crossover(space)

        # Performs the mutation
        self._mutation(space)

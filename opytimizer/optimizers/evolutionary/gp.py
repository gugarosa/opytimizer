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

        super(GP, self).__init__()

        self.p_reproduction = 0.25
        self.p_mutation = 0.1
        self.p_crossover = 0.1
        self.prunning_ratio = 0.0

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

        prunned_nodes = int(n_nodes * (1 - self.prunning_ratio))
        if prunned_nodes <= 2:
            return 2

        return prunned_nodes

    def _reproduction(self, space: TreeSpace) -> None:
        """Reproducts a number of individuals pre-selected through a tournament procedure (p. 99).

        Args:
            space: A TreeSpace object.

        """

        fitness = [agent.fit for agent in space.agents]

        n_individuals = int(space.n_agents * self.p_reproduction)

        selected = g.tournament_selection(fitness, n_individuals)
        for s in selected:
            worst = np.argmax(fitness)

            space.trees[worst] = copy.deepcopy(space.trees[s])
            space.agents[worst] = copy.deepcopy(space.agents[s])

            fitness[worst] = 0

    def _mutation(self, space: TreeSpace) -> None:
        """Mutates a number of individuals pre-selected through a tournament procedure.

        Args:
            space: A TreeSpace object.

        """

        fitness = [agent.fit for agent in space.agents]

        n_individuals = int(space.n_agents * self.p_mutation)

        selected = g.tournament_selection(fitness, n_individuals)
        for s in selected:
            n_nodes = space.trees[s].n_nodes
            if n_nodes > 1:
                max_nodes = self._prune_nodes(n_nodes)

                space.trees[s] = self._mutate(space, space.trees[s], max_nodes)
            else:
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

        mutated_tree = copy.deepcopy(tree)
        mutation_point = int(r.generate_uniform_random_number(2, max_nodes))

        sub_tree, flag = mutated_tree.find_node(mutation_point)

        # If the mutation point's parent is not a root (this may happen when the mutation point is a function),
        # and find_node() stops at a terminal node whose father is a root
        if sub_tree:
            branch = space.grow(space.min_depth, space.max_depth)

            # Checks if sub-tree should be positioned in the left
            if flag:
                sub_tree.left = branch
                branch.flag = True
            else:
                sub_tree.right = branch
                branch.flag = False

            branch.parent = sub_tree
        else:
            mutated_tree = space.grow(space.min_depth, space.max_depth)

        return mutated_tree

    def _crossover(self, space: TreeSpace) -> None:
        """Crossover a number of individuals pre-selected through a tournament procedure (p. 101).

        Args:
            space: A TreeSpace object.

        """

        fitness = [agent.fit for agent in space.agents]

        n_individuals = int(space.n_agents * self.p_crossover)
        if n_individuals % 2 != 0:
            n_individuals += 1

        selected = g.tournament_selection(fitness, n_individuals)
        for s in g.n_wise(selected):
            father_nodes = space.trees[s[0]].n_nodes
            mother_nodes = space.trees[s[1]].n_nodes

            if (father_nodes > 1) and (mother_nodes > 1):
                max_f_nodes = self._prune_nodes(father_nodes)
                max_m_nodes = self._prune_nodes(mother_nodes)

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

        father_offspring = copy.deepcopy(father)
        father_point = int(r.generate_uniform_random_number(2, max_father))

        sub_father, flag_father = father_offspring.find_node(father_point)

        mother_offspring = copy.deepcopy(mother)
        mother_point = int(r.generate_uniform_random_number(2, max_mother))

        sub_mother, flag_mother = mother_offspring.find_node(mother_point)

        if sub_father and sub_mother:
            # If father's node is positioned in the left
            if flag_father:
                branch = sub_father.left

                # If mother's node is positioned in the left
                if flag_mother:
                    sub_father.left = sub_mother.left
                    sub_mother.left.flag = True
                else:
                    sub_father.left = sub_mother.right
                    sub_mother.right.flag = True
            else:
                branch = sub_father.right

                # If mother's node is positioned in the left
                if flag_mother:
                    sub_father.right = sub_mother.left
                    sub_mother.left.flag = False
                else:
                    sub_father.right = sub_mother.right
                    sub_mother.right.flag = False

            sub_mother.parent = sub_father

            # Now, for creating the mother's offspring
            # Check if it is positioned in the left
            if flag_mother:
                sub_mother.left = branch
                branch.flag = True
            else:
                sub_mother.right = branch
                branch.flag = False

            branch.parent = sub_mother

        return father, mother

    def evaluate(self, space: Space, function: Function) -> None:
        """Evaluates the search space according to the objective function.

        Args:
            space: A TreeSpace object.
            function: A Function object that will be used as the objective function.

        """

        for (tree, agent) in zip(space.trees, space.agents):
            agent.position = copy.deepcopy(tree.position)
            agent.clip_by_bound()

            agent.fit = function(agent.position)
            if agent.fit < space.best_agent.fit:
                space.best_tree = copy.deepcopy(tree)
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: Space) -> None:
        """Wraps Genetic Programming over all trees and variables.

        Args:
            space: TreeSpace containing agents and update-related information.

        """

        self._reproduction(space)
        self._crossover(space)
        self._mutation(space)

# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Geometric Semantic Genetic Programming.

Tournament-selected mutation combines a copied subtree with a transformed random terminal.
Crossover produces one semantic offspring from two subtrees weighted by complementary terminals.
Selection and pruning parameters are inherited from GP.

References:
    A. Moraglio, K. Krawiec, and C. G. Johnson. Geometric semantic genetic programming.
    Lecture Notes in Computer Science (2012).

    G. H. de Rosa, J. P. Papa, and L. P. Papa.
    Feature selection using geometric semantic genetic programming.
    Proceedings of the Genetic and Evolutionary Computation Conference Companion (2017).

"""

import copy
from hashlib import sha1

import numpy as np

import opytimizer.math.general as g
from opytimizer.core.node import Node
from opytimizer.optimizers.evolutionary.gp import GP
from opytimizer.spaces.tree import TreeSpace


class GSGP(GP):
    """Optimize expression trees with geometric semantic variation.

    """

    def _mutation(self, space: TreeSpace) -> None:
        fitness = [agent.fit for agent in space.agents]

        n_individuals = int(space.n_agents * self.p_mutation)
        if n_individuals % 2 != 0:
            n_individuals += 1

        selected = g.tournament_selection(fitness, n_individuals)
        for s in selected:
            n_nodes = space.trees[s].n_nodes
            if n_nodes > 1:
                max_nodes = self._prune_nodes(n_nodes)
                space.trees[s] = self._mutate(space.trees[s], space.n_variables, max_nodes)

    def _mutate(self, tree: Node, n_variables: int, max_nodes: int) -> Node:
        mutated_tree = copy.deepcopy(tree)
        mutation_point = int(np.random.uniform(2, max_nodes))
        sub_tree, _ = mutated_tree.find_node(mutation_point)

        # Without a parent subtree, retain the copied tree rather than constructing a semantic branch
        if sub_tree:
            position = np.random.uniform(0.0, 1.0, n_variables)
            position_hash = sha1(repr(position).encode("ascii")).hexdigest()[:4]

            terminal = Node(position_hash, "TERMINAL", position)

            operator_id = np.random.randint(0, 3, None)
            if operator_id == 0:
                terminal.value = np.exp(terminal.value)
            elif operator_id == 1:
                terminal.value = np.fabs(np.sin(terminal.value))
            elif operator_id == 2:
                terminal.value = np.cos(np.sin(terminal.value))

            if np.random.uniform(0.0, 1.0, 1) <= 0.5:
                root = Node("SUM", "FUNCTION")
            else:
                root = Node("MUL", "FUNCTION")

            root.parent = None
            root.left = sub_tree
            root.right = terminal

            sub_tree.parent = root
            terminal.parent = root
            terminal.flag = False

            return root

        return mutated_tree

    def _crossover(self, space: TreeSpace) -> None:
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

                space.trees[s[0]] = self._cross(
                    space.trees[s[0]],
                    space.trees[s[1]],
                    space.n_variables,
                    max_f_nodes,
                    max_m_nodes,
                )

    def _cross(
        self,
        father: Node,
        mother: Node,
        n_variables: int,
        max_father: int,
        max_mother: int,
    ) -> Node:
        father_offspring = copy.deepcopy(father)
        father_point = int(np.random.uniform(2, max_father))
        sub_father, _ = father_offspring.find_node(father_point)

        mother_offspring = copy.deepcopy(mother)
        mother_point = int(np.random.uniform(2, max_mother))
        sub_mother, _ = mother_offspring.find_node(mother_point)

        if sub_father and sub_mother:
            position = np.random.uniform(0.0, 1.0, n_variables)
            position_hash = sha1(repr(position).encode("ascii")).hexdigest()[:4]

            terminal = Node(position_hash, "TERMINAL", position)
            not_terminal = Node("~" + position_hash, "TERMINAL", 1 - terminal.value)

            root = Node("SUM", "FUNCTION")
            left_node = Node("MUL", "FUNCTION")
            right_node = Node("MUL", "FUNCTION")

            root.parent = None
            root.left = left_node
            root.right = right_node

            sub_father.parent = left_node
            sub_mother.parent = right_node
            sub_mother.flag = False

            left_node.parent = root
            left_node.left = sub_father
            left_node.right = terminal

            not_terminal.parent = right_node
            terminal.parent = left_node
            terminal.flag = False

            right_node.parent = root
            right_node.left = not_terminal
            right_node.right = sub_mother
            right_node.flag = False

            return root

        return father_offspring

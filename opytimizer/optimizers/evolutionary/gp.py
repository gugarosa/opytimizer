# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Genetic Programming.

Updates require a TreeSpace and apply tournament-selected reproduction (page 99),
crossover (page 101), and subtree mutation (page 105). Pruning limits the nodes searched
for mutation and crossover. Evaluation copies bounded tree outputs into agents and records
the best tree alongside the best agent.

References:
    J. Koza. Genetic programming: On the programming of computers by means of natural selection (1992).

"""

import copy
import time
from collections.abc import Callable
from typing import Any

import numpy as np

import opytimizer.math.general as g
from opytimizer.core import Optimizer
from opytimizer.core.node import Node
from opytimizer.core.space import Space
from opytimizer.spaces.tree import TreeSpace


class GP(Optimizer):
    """Optimize expression trees with genetic programming.

    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize tree reproduction, variation, and pruning fractions.

        Args:
            params: Overrides for the supported optimizer parameters.

        Notes:
            Supported keys are ``p_reproduction`` (reproduced population fraction, 0.25),
            ``p_mutation`` (mutated population fraction, 0.1), ``p_crossover``
            (crossed population fraction, 0.1), and ``prunning_ratio``
            (fraction of nodes excluded from variation searches, 0.0).

        """

        super(GP, self).__init__()

        self.p_reproduction = 0.25
        self.p_mutation = 0.1
        self.p_crossover = 0.1
        self.prunning_ratio = 0.0

        self.build(params)

    def _prune_nodes(self, n_nodes: int) -> int:
        prunned_nodes = int(n_nodes * (1 - self.prunning_ratio))
        if prunned_nodes <= 2:
            return 2

        return prunned_nodes

    def _reproduction(self, space: TreeSpace) -> None:
        fitness = [agent.fit for agent in space.agents]

        n_individuals = int(space.n_agents * self.p_reproduction)

        selected = g.tournament_selection(fitness, n_individuals)
        for s in selected:
            worst = np.argmax(fitness)

            space.trees[worst] = copy.deepcopy(space.trees[s])
            space.agents[worst] = copy.deepcopy(space.agents[s])

            fitness[worst] = 0

    def _mutation(self, space: TreeSpace) -> None:
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
        mutated_tree = copy.deepcopy(tree)
        mutation_point = int(np.random.uniform(2, max_nodes))

        sub_tree, flag = mutated_tree.find_node(mutation_point)

        # A missing parent requires replacing the whole tree rather than attaching a subtree
        if sub_tree:
            branch = space.grow(space.min_depth, space.max_depth)

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

    def _cross(self, father: Node, mother: Node, max_father: int, max_mother: int) -> tuple[Node, Node]:
        father_offspring = copy.deepcopy(father)
        father_point = int(np.random.uniform(2, max_father))

        sub_father, flag_father = father_offspring.find_node(father_point)

        mother_offspring = copy.deepcopy(mother)
        mother_point = int(np.random.uniform(2, max_mother))

        sub_mother, flag_mother = mother_offspring.find_node(mother_point)

        if sub_father and sub_mother:
            if flag_father:
                branch = sub_father.left

                if flag_mother:
                    sub_father.left = sub_mother.left
                    sub_mother.left.flag = True
                else:
                    sub_father.left = sub_mother.right
                    sub_mother.right.flag = True
            else:
                branch = sub_father.right

                if flag_mother:
                    sub_father.right = sub_mother.left
                    sub_mother.left.flag = False
                else:
                    sub_father.right = sub_mother.right
                    sub_mother.right.flag = False

            sub_mother.parent = sub_father

            if flag_mother:
                sub_mother.left = branch
                branch.flag = True
            else:
                sub_mother.right = branch
                branch.flag = False

            branch.parent = sub_mother

        return father, mother

    def evaluate(self, space: Space, function: Callable) -> None:
        for tree, agent in zip(space.trees, space.agents):
            agent.position = copy.deepcopy(tree.position)
            agent.clip_by_bound()

            agent.fit = function(agent.position)
            if agent.fit < space.best_agent.fit:
                space.best_tree = copy.deepcopy(tree)
                space.best_agent.position = copy.deepcopy(agent.position)
                space.best_agent.fit = copy.deepcopy(agent.fit)
                space.best_agent.ts = int(time.time())

    def update(self, space: Space) -> None:
        self._reproduction(space)
        self._crossover(space)
        self._mutation(space)

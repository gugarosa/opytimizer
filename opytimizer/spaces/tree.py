"""Tree-based search space."""

import copy
from typing import List, Optional, Tuple, Union

import numpy as np

import opytimizer.utils.constant as c
from opytimizer.core import Agent, Node, Space


class TreeSpace(Space):
    """A TreeSpace class for trees, agents, variables and methods
    related to a tree-based search space.

    """

    def __init__(
        self,
        n_agents: int,
        n_variables: int,
        lower_bound: Union[float, List, Tuple, np.ndarray],
        upper_bound: Union[float, List, Tuple, np.ndarray],
        n_terminals: int = 1,
        min_depth: int = 1,
        max_depth: int = 3,
        functions: Optional[List[str]] = None,
        mapping: Optional[List[str]] = None,
    ) -> None:
        """Initialization method.

        Args:
            n_agents: Number of agents (trees).
            n_variables: Number of decision variables.
            lower_bound: Minimum possible values.
            upper_bound: Maximum possible values.
            n_terminals: Number of terminal nodes.
            min_depth: Minimum depth of the trees.
            max_depth: Maximum depth of the trees.
            functions: Function nodes.
            mapping: String-based identifiers for mapping variables' names.

        """

        n_dimensions = 1

        super().__init__(
            n_agents, n_variables, n_dimensions, lower_bound, upper_bound, mapping
        )

        if not isinstance(n_terminals, int):
            raise TypeError("`n_terminals` should be an integer")
        if n_terminals <= 0:
            raise ValueError("`n_terminals` should be > 0")
        if not isinstance(min_depth, int):
            raise TypeError("`min_depth` should be an integer")
        if min_depth <= 0:
            raise ValueError("`min_depth` should be > 0")
        if not isinstance(max_depth, int):
            raise TypeError("`max_depth` should be an integer")
        if max_depth < min_depth:
            raise ValueError("`max_depth` should be >= `min_depth`")
        if functions is None:
            functions = []
        elif not isinstance(functions, list):
            raise TypeError("`functions` should be a list")
        if any(function not in c.FUNCTION_N_ARGS for function in functions):
            raise ValueError("`functions` contains an unsupported function")

        self.n_terminals = n_terminals
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.functions = functions

        self._create_terminals()
        self._create_trees()

        self.build()

    def _create_terminals(self) -> None:
        """Creates a list of terminals."""

        self.terminals = [
            Agent(self.n_variables, self.n_dimensions, self.lb, self.ub, self.mapping)
            for _ in range(self.n_terminals)
        ]

        for terminal in self.terminals:
            terminal.fill_with_uniform()

    def _create_trees(self) -> None:
        """Creates a list of trees based on the GROW algorithm."""

        self.trees = [
            self.grow(self.min_depth, self.max_depth) for _ in range(self.n_agents)
        ]

        self.best_tree = copy.deepcopy(self.trees[0])

    def _initialize_agents(self) -> None:
        """Initializes agents with their positions and defines a best agent."""

        for agent in self.agents:
            agent.fill_with_uniform()

        self.best_agent = copy.deepcopy(self.agents[0])

    def grow(self, min_depth: int = 1, max_depth: int = 3) -> Node:
        """Creates a random tree based on the GROW algorithm.

        References:
            S. Luke. Two Fast Tree-Creation Algorithms for Genetic Programming.
            IEEE Transactions on Evolutionary Computation (2000).

        Args:
            min_depth: Minimum depth of the tree.
            max_depth: Maximum depth of the tree.

        Returns:
            (Node): Random tree based on the GROW algorithm.

        """

        if min_depth == max_depth:
            terminal_id = int(np.random.randint(0, self.n_terminals))

            return Node(terminal_id, "TERMINAL", self.terminals[terminal_id].position)

        node_id = int(np.random.randint(0, len(self.functions) + self.n_terminals))

        if node_id >= len(self.functions):
            terminal_id = node_id - len(self.functions)

            return Node(terminal_id, "TERMINAL", self.terminals[terminal_id].position)

        function_node = Node(self.functions[node_id], "FUNCTION")

        for i in range(c.FUNCTION_N_ARGS[self.functions[node_id]]):
            node = self.grow(min_depth + 1, max_depth)

            if not i:
                function_node.left = node
            else:
                function_node.right = node
                node.flag = False

            node.parent = function_node

        return function_node

"""Cell.
"""

import copy
from typing import Generator, Tuple

import networkx as nx
from networkx import DiGraph

from opytimizer.core import InnerBlock, InputBlock, OutputBlock
from opytimizer.core.block import Block


class Cell(DiGraph):
    """A Cell serves a Direct Acyclic Graph (DAG) which holds blocks as nodes and edges that
    connects operation paths between the nodes.

    """

    def __init__(self, blocks: Block, edges: Tuple[Block, Block]) -> None:
        """Initialization method.

        Args:
            type: Type of the block.
            pointer: Any type of callable to be applied when block is called.

        """

        super().__init__()

        # Iterates through all possible blocks
        for i, block in enumerate(blocks):
            # Adds block identifier to the DAG
            # We also add a `block` key to retain object's information
            self.add_node(i, block=block)

        # Iterates through all possible edges
        for (u, v) in edges:
            # Checks if nodes actually exists in DAG
            if u in self.nodes and v in self.nodes:
                # Checks if `u` number of outputs matches the number of `v` inputs
                if self.nodes[u]["block"].n_output == self.nodes[v]["block"].n_input:
                    # Adds edge to the DAG
                    self.add_edge(u, v)

    def __call__(self, *args) -> Generator:
        """Performs a forward pass over the cell.

        Returns:
            (Generator): Output for each possible path in DAG.

        """

        # Checks whether current DAG is valid or not
        if not self.valid:
            # If not valid, it should not be forwarded and should return empty outputs
            return []

        # Gathers a list of paths from `input` to `output`
        paths = list(nx.all_simple_paths(self, self.input_idx, self.output_idx))

        # Iterates through all paths
        outputs = []
        for path in paths:
            # Makes a copy of current arguments
            current_args = copy.deepcopy(args)

            # Iterates through all nodes in path
            for node in path:
                # Passes down through every node in path
                current_args = self.nodes[node]["block"](*current_args)

                # Makes sure that outputs are always encoded as tuples
                if type(current_args) != tuple:
                    current_args = (current_args,)

            # Appends the outputs
            outputs.append(current_args)

        return outputs

    @property
    def input_idx(self) -> int:
        """Index of the input node."""

        for node in self.nodes:
            if self.nodes[node]["block"].type == "input":
                return node

        return -1

    @property
    def output_idx(self) -> int:
        """Index of the output node."""

        for node in self.nodes:
            if self.nodes[node]["block"].type == "output":
                return node

        return -1

    @property
    def valid(self) -> bool:
        """Whether cell is valid or not."""

        if self.input_idx == -1 or self.output_idx == -1:
            return False

        return nx.is_directed_acyclic_graph(self)


if __name__ == "__main__":
    import numpy as np

    def f1(x, y):
        return x, y

    def f2(x, y):
        return x + 1, y + 1

    cell = Cell(
        [
            InputBlock(2, 2),
            InnerBlock(f1, 2, 2),
            InnerBlock(f2, 2, 2),
            OutputBlock(2, 2),
        ],
        [(0, 1), (1, 3), (0, 2), (2, 3)],
    )
    x = 1
    y = 2

    print(cell.edges)

    # print(cell.nodes[0]['block'].n_input)

    outputs = cell(x, y)

    print(outputs)

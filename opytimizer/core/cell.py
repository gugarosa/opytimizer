"""Cell.
"""


import networkx as nx
from networkx import DiGraph

from opytimizer.core import Block


class Cell(DiGraph):
    """A Cell serves a Direct Acyclic Graph (DAG) which holds blocks as nodes and edges that
    connects operation paths between the nodes.

    """

    def __init__(self, blocks, edges):
        """Initialization method.

        Args:
            type (str): Type of the block.
            pointer (callable): Any type of callable to be applied when block is called.

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
                # Adds edge to the DAG
                self.add_edge(u, v)

    @property
    def n_blocks(self):
        """int: Number of blocks.

        """

        return len(self.nodes)

    @property
    def input_idx(self):
        """int: Index of the input node.

        """

        for node in self.nodes:
            if self.nodes[node]['block'].type == 'input':
                return node
        
        return -1

    @property
    def output_idx(self):
        """int: Index of the output node.

        """

        for node in self.nodes:
            if self.nodes[node]['block'].type == 'output':
                return node

        return -1

    @property
    def valid(self):
        """bool: Whether cell is valid or not.

        """

        if self.input_idx == -1 or self.output_idx == -1:
            return False

        return nx.is_directed_acyclic_graph(self)

    def forward(self, x):
        """
        """
        
        # Checks whether current DAG is valid or not
        if not self.valid:
            # If not valid, it should not be forwarded and should return empty outputs
            return []

        paths = list(nx.all_simple_paths(self, self.input_idx, self.output_idx))

        outputs = []
        for path in paths:
            print(path)
            x_path = x
            for node in path:
                x_path = self.nodes[node]['block'](x_path)
            outputs.append(x_path)

        return outputs




if __name__ == '__main__':
    cell = Cell([Block('input', lambda y: y),
                 Block('intermediate', lambda y: y),
                 Block('intermediate', lambda y: y),
                 Block('intermediate', lambda y: y),
                 Block('output', lambda y: y)],
                 [(0, 1), (1, 2), (2, 4), (2, 3), (3, 4), (0, 2)])
    x = 1

    # print(cell.nodes)
    # print(cell.edges)
    print(cell.forward(x))

    # print(cell.valid)
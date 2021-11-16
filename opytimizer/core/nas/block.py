"""Block.
"""

import networkx as nx
from networkx import DiGraph

from opytimizer.core.nas.cell import Cell

import opytimizer.utils.constant as c


class Block(DiGraph):
    """
    """

    def __init__(self, cells, edges):
        """
        """

        super().__init__()

        # Iterates through all possible cells
        for i, cell in enumerate(cells):
            # Adds cell identifier to the DAG
            # We also add a `cell` key to retain object's information
            self.add_node(i, cell=cell)
        
        # Iterates through all possible edges
        for (u, v) in edges:
            # Adds edge to the DAG
            self.add_edge(u, v)

        # Remaining properties
        self.fit = c.FLOAT_MAX

    @property
    def n_cells(self):
        return len(self.nodes)

    @property
    def valid(self):
        return nx.is_directed_acyclic_graph(self)

    @property
    def input_idx(self):
        for node in b.nodes:
            if b.nodes[node]['cell'].type == 'input':
                return node

    @property
    def output_idx(self):
        for node in b.nodes:
            if b.nodes[node]['cell'].type == 'output':
                return node

    def forward(self):
        """
        """

        paths = list(nx.all_simple_paths(self, self.input_idx, self.output_idx))

        outputs = []
        for path in paths:
            print(path)     




if __name__ == '__main__':
    b = Block([Cell('input'), Cell('layer'), Cell('layer'), Cell('output')], [(0, 1), (1, 2), (0, 4), (1, 3), (3, 4)])

    b.forward()
# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

from opytimizer.spaces.tree import TreeSpace

n_agents = 2
n_variables = 5
n_terminals = 2

min_depth = 2
max_depth = 5

func_nodes = ["SUM", "SUB", "MUL", "DIV"]

lower_bound = [0.1, 0.3, 0.5, 0.7, 0.9]
upper_bound = [0.2, 0.4, 0.6, 0.8, 1.0]

s = TreeSpace(
    n_agents,
    n_variables,
    lower_bound,
    upper_bound,
    n_terminals,
    min_depth,
    max_depth,
    func_nodes,
)

print(s.trees[0])
print(f"Position: {s.trees[0].position}")
print(f"\nPre Order: {s.trees[0].pre_order}")
print(f"\nPost Order: {s.trees[0].post_order}")
print(
    f"\nNodes: {s.trees[0].n_nodes} | Leaves: {s.trees[0].n_leaves} | "
    f"Minimum Depth: {s.trees[0].min_depth} | Maximum Depth: {s.trees[0].max_depth}"
)

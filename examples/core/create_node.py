# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opytimizer.core.node import Node

n1 = Node(name="0", category="TERMINAL", value=np.array(1))
n2 = Node(name="1", category="TERMINAL", value=np.array(2))

print(n1)
print(f"Post Order: {n1.post_order} | Size: {n1.n_nodes}.")

t = Node(name="SUM", category="FUNCTION", left=n1, right=n2)

# Parent links are explicit because the constructor retains references as supplied
n1.parent = t
n2.parent = t

print(t)
print(f"Post Order: {t.post_order} | Size: {t.n_nodes} | Minimum Depth: {t.min_depth} | Maximum Depth: {t.max_depth}.")

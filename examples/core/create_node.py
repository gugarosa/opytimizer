from opytimizer.core.node import Node

# Creating two new Nodes
n1 = Node(name='Node1', type='LEAF', value=1)
n2 = Node(name='Node2', type='LEAF', value=2)

# Outputting information about one of the nodes
print(n1)
print(f'Post Order: {n1.post_order} | Size: {n1.n_nodes}.')

# Additionally, one can stack nodes to create a tree
t = Node(name='Tree', type='ROOT', value=0, left=n1, right=n2)

# Defining `n1` and `n2` parent as `t`
n1.parent = t
n2.parent = t

# Outputting information about the tree
print(t)
print(f'Post Order: {t.post_order} | Size: {t.n_nodes} | Minimum Depth: {t.min_depth} | Maximum Depth: {t.max_depth}.')

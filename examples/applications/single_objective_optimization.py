import numpy as np
from opytimark.markers.n_dimensional import Sphere

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm.fa import FA
from opytimizer.spaces import SearchSpace

from opytimizer.utils.callback import SnapshotCallback

# Random seed for experimental consistency
np.random.seed(0)

# Number of agents and decision variables
n_agents = 20
n_variables = 2

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = FA()
function = Function(Sphere())

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, store_best_only=True)

# Runs the optimization task
opt.start(n_iterations=100, callbacks=[SnapshotCallback(iterations_per_snapshot=10)])
opt.start(n_iterations=100, callbacks=[SnapshotCallback(iterations_per_snapshot=10)])

opt.save('out.pkl')
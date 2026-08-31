import numpy as np

from opytimizer import Opytimizer
from opytimizer.optimizers.misc.nds import NDS
from opytimizer.spaces import ParetoSpace

# Random seed for experimental consistency
np.random.seed(0)

# Defines the number of points `n` and the number of objectives `k`
n_points = 100
n_objectives = 3

# Defines the agents to be initialized within the ParetoSpace
# Note they are a multi-dimensional vector of shape [n, k],
data_points = np.random.uniform(size=(n_points, n_objectives))

# Creates the space and optimizer
space = ParetoSpace(data_points)
optimizer = NDS()

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, lambda _: 0, save_agents=False)

# Runs the optimization task
opt.start()

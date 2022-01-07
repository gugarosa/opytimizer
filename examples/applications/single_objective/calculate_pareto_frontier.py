import numpy as np

import opytimizer.math.random as r
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.misc.nds import NDS
from opytimizer.spaces import ParetoSpace

# Random seed for experimental consistency
np.random.seed(0)

# Defines the number of points `n` and the number of objectives `k`
n_points = 100
n_objectives = 3

# Defines the agents to be initialized within the ParetoSpace
# Note they are a multi-dimensional vector of shape [n, k],
data_points = r.generate_uniform_random_number(size=(n_points, n_objectives))

# Creates the space, optimizer and function
space = ParetoSpace(data_points)
optimizer = NDS()
function = Function(lambda x: 0)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=False)

# Runs the optimization task
opt.start()

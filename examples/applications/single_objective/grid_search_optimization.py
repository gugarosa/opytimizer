from opytimark.markers.n_dimensional import Sphere

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.misc import GS
from opytimizer.spaces import GridSpace

# Number of decision variables and step size of the grid
n_variables = 2
step = [0.1, 1]

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Creates the space, optimizer and function
space = GridSpace(n_variables, step, lower_bound, upper_bound)
optimizer = GS()
function = Function(Sphere())

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, store_only_best_agent=True)

# Runs the optimization task
opt.start()

from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.math import benchmark
from opytimizer.optimizers.misc.gs import GS
from opytimizer.spaces.grid import GridSpace

# Number of decision variables
n_variables = 2

# And also the size of the step in the grid
step = 0.1

# Lower and upper bounds (has to be the same size as n_variables)
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Creating the GridSpace class
s = GridSpace(n_variables=n_variables, step=step,
              lower_bound=lower_bound, upper_bound=upper_bound)

# Creating GS optimizer
p = GS()

# Creating Function's object
f = Function(pointer=benchmark.sphere)

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

# Running the optimization task
history = o.start()

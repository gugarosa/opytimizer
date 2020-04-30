from opytimizer.spaces.grid import GridSpace

# We need to define the number of decision variables
n_variables = 2

# And also the size of the step in the grid
step = 0.1

# Finally, we define the lower and upper bounds
# Note that they have to be the same size as n_variables
lower_bound = [0.5, 1]
upper_bound = [2.0, 2]

# Creating the GridSpace object
s = GridSpace(n_variables=n_variables, step=step,
              lower_bound=lower_bound, upper_bound=upper_bound)

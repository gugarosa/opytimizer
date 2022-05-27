from opytimizer.core import Agent

# We need to define the amount of decision variables
# and its dimension (single, complex, quaternion, octonion, sedenion)
n_variables = 2
n_dimensions = 2

# We also need to define its bounds
lower_bound = [0, 0]
upper_bound = [1, 1]

# Creates a new Agent
a = Agent(n_variables, n_dimensions, lower_bound, upper_bound)

# Prints out some properties
print(a.n_variables, a.n_dimensions)
print(a.position, a.fit)
print(a.mapped_position)

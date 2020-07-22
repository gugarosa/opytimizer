from opytimizer.core.agent import Agent

# We need to define the amount of decision variables
# and its dimension (single, complex, quaternion, octonion, sedenion)
n_variables = 1
n_dimensions = 2

# Creating a new Agent
a = Agent(n_variables=n_variables, n_dimensions=n_dimensions)

# Printing out some agent's properties
print(a.n_variables, a.n_dimensions)
print(a.position, a.fit)

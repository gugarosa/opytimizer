from opytimizer.core.agent import Agent

# We need to define the amount of decision variables
# and its dimension (single, complex, quaternion, octonion)
n_variables = 1
n_dimensions = 2

# Creating a new Agent
a = Agent(n_variables=n_variables, n_dimensions=n_dimensions)

# Printing out agent's object
print(a)

from opytimizer.spaces import GridSpace

# Define the number of decision variables
n_variables = 2

# Also defines the step size of each variable
# and their corresponding lower and upper bounds
step = [0.1, 1]
lower_bound = [0.5, 1]
upper_bound = [2.0, 2]

# Creating the GridSpace object
s = GridSpace(n_variables, step, lower_bound, upper_bound)

# Prints out some properties
print(s.n_agents, s.n_variables)
print(s.agents, s.best_agent)
print(s.best_agent.position)

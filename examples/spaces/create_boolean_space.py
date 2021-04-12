from opytimizer.spaces.boolean import BooleanSpace

# We need to define the number of agents, decision variables and iterations
n_agents = 2
n_variables = 5
n_iterations = 10

# Creating the BooleanSpace object
s = BooleanSpace(n_agents=n_agents, n_variables=n_variables, n_iterations=n_iterations)

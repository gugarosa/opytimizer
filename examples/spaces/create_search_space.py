from opytimizer.spaces.search import SearchSpace

# Define the number of agents and decision variables
n_agents = 2
n_variables = 5

# Also defines the corresponding lower and upper bounds
# Note that they have to be the same size as `n_variables`
lower_bound = [0.1, 0.3, 0.5, 0.7, 0.9]
upper_bound = [0.2, 0.4, 0.6, 0.8, 1.0]

# Creates the SearchSpace
s = SearchSpace(n_agents=n_agents, n_variables=n_variables,
                lower_bound=lower_bound, upper_bound=upper_bound)

# Prints out some properties
print(s.n_agents, s.n_variables)
print(s.agents, s.best_agent)
print(s.best_agent.position)

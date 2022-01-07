import opytimizer.math.random as r
from opytimizer.spaces import ParetoSpace

# Defines the number of points `n` and the number of objectives `k`
n_points = 10
n_objectives = 3

# Defines the agents to be initialized within the ParetoSpace
# Note they are a multi-dimensional vector of shape [n, k],
data_points = r.generate_uniform_random_number(size=(n_points, n_objectives))

# Creates the ParetoSpace
s = ParetoSpace(data_points)

# Prints out some properties
print(s.n_agents, s.n_variables)
print(s.agents, s.best_agent)
print(s.best_agent.position)

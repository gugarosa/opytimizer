from opytimizer.spaces import ParetoSpace

import opytimizer.math.random as r

# Defines the number of points `n` and the number of objectives `k`
n_points = 10
n_objectives = 3

# Defines the agents to be initialized within the ParetoSpace
# Note they are a multi-dimensional vector of shape [n, k],
data_points = r.generate_uniform_random_number(size=(n_points, n_objectives))

# Creates the ParetoSpace
s = ParetoSpace(data_points)

for a in s.agents:
    print(a.position)
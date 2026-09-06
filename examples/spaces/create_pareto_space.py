# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opytimizer.spaces import ParetoSpace

n_points = 10
n_objectives = 3

# Each row stores one candidate's objective values for Pareto ranking
data_points = np.random.uniform(size=(n_points, n_objectives))

s = ParetoSpace(data_points)

print(s.n_agents, s.n_variables)
print(s.agents, s.best_agent)
print(s.best_agent.position)

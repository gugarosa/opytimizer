# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

from opytimizer.spaces import GridSpace

n_variables = 2

step = [0.1, 1]
lower_bound = [0.5, 1]
upper_bound = [2.0, 2]

s = GridSpace(n_variables, step, lower_bound, upper_bound)

print(s.n_agents, s.n_variables)
print(s.agents, s.best_agent)
print(s.best_agent.position)

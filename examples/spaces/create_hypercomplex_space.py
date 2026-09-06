# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

from opytimizer.spaces import HyperComplexSpace

n_agents = 2
n_variables = 5
n_dimensions = 4

s = HyperComplexSpace(n_agents=n_agents, n_variables=n_variables, n_dimensions=n_dimensions)

print(s.n_agents, s.n_variables, s.n_dimensions)
print(s.agents, s.best_agent)
print(s.best_agent.position)

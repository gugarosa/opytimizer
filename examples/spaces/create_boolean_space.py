# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

from opytimizer.spaces import BooleanSpace

n_agents = 2
n_variables = 5

s = BooleanSpace(n_agents, n_variables)

print(s.n_agents, s.n_variables)
print(s.agents, s.best_agent)
print(s.best_agent.position)

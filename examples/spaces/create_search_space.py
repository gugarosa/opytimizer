# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

from opytimizer.spaces.search import SearchSpace

n_agents = 2
n_variables = 5

lower_bound = [0.1, 0.3, 0.5, 0.7, 0.9]
upper_bound = [0.2, 0.4, 0.6, 0.8, 1.0]

s = SearchSpace(
    n_agents=n_agents,
    n_variables=n_variables,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
)

print(s.n_agents, s.n_variables)
print(s.agents, s.best_agent)
print(s.best_agent.position)

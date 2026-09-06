# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

from opytimizer.core import Agent

# Dimensions 1, 2, 4, 8, and 16 represent real, complex, quaternion, octonion, and sedenion values
n_variables = 2
n_dimensions = 2

lower_bound = [0, 0]
upper_bound = [1, 1]

a = Agent(n_variables, n_dimensions, lower_bound, upper_bound)

print(a.n_variables, a.n_dimensions)
print(a.position, a.fit)
print(a.mapped_position)

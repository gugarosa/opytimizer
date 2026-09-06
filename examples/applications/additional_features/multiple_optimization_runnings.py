# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace


def sphere(x: np.ndarray) -> float:
    """Evaluate the sphere objective.

    Args:
        x: Candidate position array.

    Returns:
        Sum of squared decision variables.

    """

    return np.sum(x**2)


# Random seed for experimental consistency
np.random.seed(0)

n_agents = 20
n_variables = 2

lower_bound = [-10, -10]
upper_bound = [10, 10]

space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()

opt = Opytimizer(space, optimizer, sphere, save_agents=False)

# Standard PSO retains its state between calls, giving 100 total iterations here
# Other algorithms may restart iteration-local schedules on each call
opt.start(n_iterations=50)
opt.start(n_iterations=25)
opt.start(n_iterations=25)

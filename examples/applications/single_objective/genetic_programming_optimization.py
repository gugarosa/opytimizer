# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opytimizer import Opytimizer
from opytimizer.optimizers.evolutionary import GP
from opytimizer.spaces import TreeSpace


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
n_terminals = 2
n_variables = 2

min_depth = 2
max_depth = 5

functions = ["SUM", "MUL", "DIV"]
lower_bound = [-10, -10]
upper_bound = [10, 10]

space = TreeSpace(
    n_agents,
    n_variables,
    lower_bound,
    upper_bound,
    n_terminals,
    min_depth,
    max_depth,
    functions,
)
optimizer = GP()

opt = Opytimizer(space, optimizer, sphere, save_agents=False)

opt.start(n_iterations=1000)

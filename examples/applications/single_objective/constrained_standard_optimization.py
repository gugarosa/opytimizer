# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opytimizer import Opytimizer
from opytimizer.functions import ConstrainedFunction
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


def c_1(x: np.ndarray) -> np.ndarray:
    """Check whether the sum of the first two variables is negative.

    Args:
        x: Two-variable candidate position array.

    Returns:
        Single-element boolean array indicating constraint validity.

    """

    return x[0] + x[1] < 0


# Random seed for experimental consistency
np.random.seed(0)

n_agents = 20
n_variables = 2

lower_bound = [-10, -10]
upper_bound = [10, 10]

space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()
function = ConstrainedFunction(sphere, [c_1], penalty=100.0)

opt = Opytimizer(space, optimizer, function, save_agents=False)

opt.start(n_iterations=1000)

# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opytimizer import Opytimizer
from opytimizer.functions.multi_objective import MultiObjectiveWeightedFunction
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace


def rastrigin(x: np.ndarray) -> float:
    """Evaluate the Rastrigin objective.

    Args:
        x: Candidate position array.

    Returns:
        Sum of quadratic and periodic penalties.

    """

    return 10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


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
function = MultiObjectiveWeightedFunction([rastrigin, sphere], [0.5, 0.5])

opt = Opytimizer(space, optimizer, function, save_agents=False)

opt.start(n_iterations=1000)

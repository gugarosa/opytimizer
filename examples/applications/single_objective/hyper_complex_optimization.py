# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np

import opytimizer.math.hyper as h
from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import HyperComplexSpace

# Random seed for experimental consistency
np.random.seed(0)

n_agents = 20
n_variables = 2
n_dimensions = 4

lower_bound = [-10, -10]
upper_bound = [10, 10]


@h.span_to_hyper_value(lower_bound, upper_bound)
def sphere(x: np.ndarray) -> float:
    """Evaluate the sphere objective after spanning hypercomplex values to bounds.

    Args:
        x: Spanned real-valued decision variables supplied by the decorator.

    Returns:
        Sum of squared decision variables.

    """

    return np.sum(x**2)


space = HyperComplexSpace(n_agents, n_variables, n_dimensions)
optimizer = PSO()

opt = Opytimizer(space, optimizer, sphere, save_agents=False)

opt.start(n_iterations=1000)

# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opytimizer import Opytimizer
from opytimizer.optimizers.misc import GS
from opytimizer.spaces import GridSpace


def sphere(x: np.ndarray) -> float:
    """Evaluate the sphere objective.

    Args:
        x: Candidate position array.

    Returns:
        Sum of squared decision variables.

    """

    return np.sum(x**2)


n_variables = 2
step = [0.1, 1]

lower_bound = [-10, -10]
upper_bound = [10, 10]

space = GridSpace(n_variables, step, lower_bound, upper_bound)
optimizer = GS()

opt = Opytimizer(space, optimizer, sphere, save_agents=False)

opt.start()

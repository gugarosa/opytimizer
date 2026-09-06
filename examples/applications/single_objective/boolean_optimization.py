# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opytimizer import Opytimizer
from opytimizer.optimizers.boolean import BPSO
from opytimizer.spaces import BooleanSpace

values = np.array([55, 10, 47, 5, 4])
weights = np.array([95, 4, 60, 32, 23])


def knapsack(x: np.ndarray) -> float:
    """Minimize the negated value of a feasible knapsack selection.

    Args:
        x: Binary selection array with one row per item.

    Returns:
        Negative selected value, or the maximum float when capacity is exceeded.

    """

    selected = x.ravel()
    if weights @ selected > 100:
        return np.finfo(float).max
    return -(values @ selected)


# Random seed for experimental consistency
np.random.seed(0)

n_agents = 5
n_variables = 5

params = {
    "c1": np.random.randint(0, 2, size=(n_variables, 1)),
    "c2": np.random.randint(0, 2, size=(n_variables, 1)),
}

space = BooleanSpace(n_agents, n_variables)
optimizer = BPSO(params)

opt = Opytimizer(space, optimizer, knapsack, save_agents=False)

opt.start(n_iterations=1000)

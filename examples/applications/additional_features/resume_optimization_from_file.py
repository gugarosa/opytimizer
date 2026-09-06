# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace
from opytimizer.utils.callback import CheckpointCallback


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

opt.start(n_iterations=10, callbacks=[CheckpointCallback(frequency=10)])

del opt

# This deterministic objective and standard PSO can continue the same search
# Other algorithms may restart iteration-local schedules when start() is called
# Load only trusted checkpoints because deserialization can execute code
opt = Opytimizer.load("iter_10_checkpoint.pkl")
opt.start(n_iterations=25)

# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opytimizer import Opytimizer
from opytimizer.optimizers.misc.nds import NDS
from opytimizer.spaces import ParetoSpace

# Random seed for experimental consistency
np.random.seed(0)

n_points = 100
n_objectives = 3

# Each row is one candidate's objective vector, not a decision-variable position
data_points = np.random.uniform(size=(n_points, n_objectives))

space = ParetoSpace(data_points)
optimizer = NDS()

# NDS ranks the supplied objective vectors without calling the objective
opt = Opytimizer(space, optimizer, lambda _: 0, save_agents=False)

opt.start()

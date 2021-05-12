import numpy as np

from opytimizer.optimizers.boolean.bpso import BPSO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'w': np.array([1]),
    'c1': np.array([0]),
    'c2': np.array([1])
}

# Creates a BPSO optimizer
o = BPSO(params=params)

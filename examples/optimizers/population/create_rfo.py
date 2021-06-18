import numpy as np

import opytimizer.math.random as r
from opytimizer.optimizers.population import RFO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'phi': r.generate_uniform_random_number(0, 2*np.pi)[0],
    'theta': r.generate_uniform_random_number()[0],
    'p_replacement': 0.05
}

# Creates a RFO optimizer
o = RFO(params=params)

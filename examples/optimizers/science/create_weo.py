import numpy as np

from opytimizer.optimizers.science import WEO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    "E_min": -3.5,
    "E_max": -0.5,
    "theta_min": -np.pi / 3.6,
    "theta_max": -np.pi / 9,
}

# Creates a WEO optimizer
o = WEO(params=params)

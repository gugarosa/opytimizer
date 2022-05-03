import numpy as np

from opytimizer.optimizers.science import AIG

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"alpha": np.pi, "beta": np.pi}

# Creates an AIG optimizer
o = AIG(params=params)

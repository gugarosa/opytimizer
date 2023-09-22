from opytimizer.optimizers.science import SMA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"z": 0.03}

# Creates a WCA optimizer
o = SMA(params=params)

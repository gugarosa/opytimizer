from opytimizer.optimizers.science import WCA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"nsr": 10, "d_max": 0.1}

# Creates a WCA optimizer
o = WCA(params=params)

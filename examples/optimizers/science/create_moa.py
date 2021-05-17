from opytimizer.optimizers.science import MOA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'alpha': 1.0,
    'rho': 2.0
}

# Creates a MOA optimizer
o = MOA(params=params)

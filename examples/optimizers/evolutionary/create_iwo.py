from opytimizer.optimizers.evolutionary.iwo import IWO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'min_seeds': 0,
    'max_seeds': 5,
    'e': 2,
    'init_sigma': 3,
    'final_sigma': 0.001
}

# Creates an IWO optimizer
o = IWO(params=params)

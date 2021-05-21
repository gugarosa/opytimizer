from opytimizer.optimizers.evolutionary import RRA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'd_runner': 2,
    'd_root': 0.01,
    'tol': 0.01,
    'max_stall': 1000
}

# Creates an RRA optimizer
o = RRA(params=params)

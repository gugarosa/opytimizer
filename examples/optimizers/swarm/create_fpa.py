from opytimizer.optimizers.swarm import FPA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'beta': 1.5,
    'eta': 0.2,
    'p': 0.8
}

# Creates a FPA optimizer
o = FPA(params=params)

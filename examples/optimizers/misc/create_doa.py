from opytimizer.optimizers.misc import DOA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'r': 1.0
}

# Creates a DOA optimizer
o = DOA(params=params)

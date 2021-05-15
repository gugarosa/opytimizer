from opytimizer.optimizers.evolutionary import GOGHS

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'pm': 0.1
}

# Creates a GOGHS optimizer
o = GOGHS(params=params)

from opytimizer.optimizers.evolutionary.es import ES

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'child_ratio': 0.5
}

# Creates an ES optimizer
o = ES(params=params)

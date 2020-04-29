from opytimizer.optimizers.evolutionary.es import ES

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'child_ratio': 0.5
}

# Creating an ES optimizer
o = ES(hyperparams=hyperparams)

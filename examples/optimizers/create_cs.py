from opytimizer.optimizers.cs import CS

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'alpha': 0.3,
    'beta': 1.5,
    'p': 0.2
}

# Creating a CS optimizer
o = CS(hyperparams=hyperparams)

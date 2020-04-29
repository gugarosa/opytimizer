from opytimizer.optimizers.evolutionary.de import DE

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'CR': 0.9,
    'F': 0.7
}

# Creating a DE optimizer
o = DE(hyperparams=hyperparams)

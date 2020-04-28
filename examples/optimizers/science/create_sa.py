from opytimizer.optimizers.science.sa import SA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'T': 100,
    'beta': 0.999
}

# Creating a SA optimizer
o = SA(hyperparams=hyperparams)

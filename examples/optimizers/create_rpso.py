from opytimizer.optimizers.rpso import RPSO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'c1': 1.7,
    'c2': 1.7
}

# Creating an RPSO optimizer
o = RPSO(hyperparams=hyperparams)

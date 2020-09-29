from opytimizer.optimizers.misc.doa import DOA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'r': 1.0
}

# Creating a DOA optimizer
o = DOA(hyperparams=hyperparams)

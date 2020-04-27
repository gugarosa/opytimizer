from opytimizer.optimizers.ssdo import SSDO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'c': 2.0
}

# Creating an SSDO optimizer
o = SSDO(hyperparams=hyperparams)

from opytimizer.optimizers.mrfo import MRFO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    's': 2.0
}

# Creating a MRFO optimizer
o = MRFO(hyperparams=hyperparams)

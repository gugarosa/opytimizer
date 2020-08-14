from opytimizer.optimizers.evolutionary.hs import GOGHS

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'pm': 0.1
}

# Creating a GOGHS optimizer
o = GOGHS(hyperparams=hyperparams)

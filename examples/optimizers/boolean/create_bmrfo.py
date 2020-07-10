from opytimizer.optimizers.boolean.bmrfo import BMRFO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'S': 1
}

# Creating a BMRFO optimizer
o = BMRFO(hyperparams=hyperparams)

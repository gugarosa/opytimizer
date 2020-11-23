from opytimizer.optimizers.swarm.goa import GOA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'c_min': 0.00001,
    'c_max': 1,
    'f': 0.5,
    'l': 1.5
}

# Creating a GOA optimizer
o = GOA(hyperparams=hyperparams)

from opytimizer.optimizers.swarm.ba import BA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'f_min': 0,
    'f_max': 2,
    'A': 0.5,
    'r': 0.5
}

# Creating a BA optimizer
o = BA(hyperparams=hyperparams)

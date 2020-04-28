from opytimizer.optimizers.swarm.fpa import FPA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'beta': 1.5,
    'eta': 0.2,
    'p': 0.8
}

# Creating a FPA optimizer
o = FPA(hyperparams=hyperparams)

from opytimizer.optimizers.swarm.js import NBJS

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'eta': 4.0,
    'beta': 3.0,
    'gamma': 0.1
}

# Creating a NBJS optimizer
o = NBJS(hyperparams=hyperparams)

from opytimizer.optimizers.swarm.mrfo import MRFO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'S': 2.0
}

# Creating an MRFO optimizer
o = MRFO(hyperparams=hyperparams)

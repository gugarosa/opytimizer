from opytimizer.optimizers.swarm.mfo import MFO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'b': 1
}

# Creating a MFO optimizer
o = MFO(hyperparams=hyperparams)

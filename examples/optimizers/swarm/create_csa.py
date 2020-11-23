from opytimizer.optimizers.swarm.csa import CSA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'fl': 2.0,
    'AP': 0.1
}

# Creating a CSA optimizer
o = CSA(hyperparams=hyperparams)

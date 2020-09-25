from opytimizer.optimizers.swarm.boa import BOA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'c': 0.01,
    'a': 0.1,
    'p': 0.8
}

# Creating a BOA optimizer
o = BOA(hyperparams=hyperparams)

from opytimizer.optimizers.swarm.sfo import SFO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
hyperparams = {
    'PP': 0.1,
    'A': 4,
    'e': 0.001
}

# Creating a SFO optimizer
o = SFO(hyperparams=hyperparams)

from opytimizer.optimizers.swarm import ABC

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"n_trials": 10}

# Creates an ABC optimizer
o = ABC(params=params)

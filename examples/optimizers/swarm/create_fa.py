from opytimizer.optimizers.swarm import FA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"alpha": 0.5, "beta": 0.2, "gamma": 1.0}

# Creates a FA optimizer
o = FA(params=params)

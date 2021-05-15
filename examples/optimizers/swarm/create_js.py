from opytimizer.optimizers.swarm import JS

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'eta': 4.0,
    'beta': 3.0,
    'gamma': 0.1
}

# Creates a JS optimizer
o = JS(params=params)

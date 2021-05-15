from opytimizer.optimizers.swarm import PSO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'w': 0.7,
    'c1': 1.7,
    'c2': 1.7
}

# Creates a PSO optimizer
o = PSO(params=params)

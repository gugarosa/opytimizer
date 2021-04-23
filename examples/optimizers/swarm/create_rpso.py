from opytimizer.optimizers.swarm.pso import RPSO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'c1': 1.7,
    'c2': 1.7
}

# Creating an RPSO optimizer
o = RPSO(params=params)

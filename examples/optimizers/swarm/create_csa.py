from opytimizer.optimizers.swarm.csa import CSA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'fl': 2.0,
    'AP': 0.1
}

# Creates a CSA optimizer
o = CSA(params=params)

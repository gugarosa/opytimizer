from opytimizer.optimizers.swarm.sbo import SBO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'alpha': 0.9,
    'p_mutation': 0.05,
    'z': 0.02
}

# Creates a SBO optimizer
o = SBO(params=params)

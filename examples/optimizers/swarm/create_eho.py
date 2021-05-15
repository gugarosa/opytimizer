from opytimizer.optimizers.swarm import EHO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'alpha': 0.5,
    'beta': 0.1,
    'n_clans': 10
}

# Creates an EHO optimizer
o = EHO(params=params)

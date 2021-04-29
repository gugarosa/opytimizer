from opytimizer.optimizers.swarm.abo import ABO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'sunspot_ratio': 0.9,
    'a': 2.0
}

# Creates an ABO optimizer
o = ABO(params=params)

from opytimizer.optimizers.swarm.sca import SCA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'r_min': 0,
    'r_max': 2,
    'a': 3
}

# Creates a SCA optimizer
o = SCA(params=params)

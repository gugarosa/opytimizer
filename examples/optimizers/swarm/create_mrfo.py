from opytimizer.optimizers.swarm import MRFO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'S': 2.0
}

# Creates an MRFO optimizer
o = MRFO(params=params)

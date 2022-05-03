from opytimizer.optimizers.swarm import STOA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"Cf": 2.0, "u": 1.0, "v": 1.0}

# Creates an STOA optimizer
o = STOA(params=params)

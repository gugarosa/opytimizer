from opytimizer.optimizers.swarm import AIWPSO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"w": 0.7, "w_min": 0.1, "w_max": 0.9, "c1": 1.5, "c2": 2}

# Creates an AIWPSO optimizer
o = AIWPSO(params=params)

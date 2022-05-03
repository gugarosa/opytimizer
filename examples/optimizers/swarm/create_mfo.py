from opytimizer.optimizers.swarm import MFO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"b": 1}

# Creates a MFO optimizer
o = MFO(params=params)

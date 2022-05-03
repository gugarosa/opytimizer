from opytimizer.optimizers.swarm import FSO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"beta": 0.5}

# Creates a FSO optimizer
o = FSO(params=params)

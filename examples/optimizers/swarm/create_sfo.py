from opytimizer.optimizers.swarm import SFO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"PP": 0.1, "A": 4, "e": 0.001}

# Creates a SFO optimizer
o = SFO(params=params)

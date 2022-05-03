from opytimizer.optimizers.swarm import AF

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"c1": 0.75, "c2": 1.25, "m": 10, "Q": 0.75}

# Creates an AF optimizer
o = AF(params=params)

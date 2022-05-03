from opytimizer.optimizers.population import OSA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"beta": 1.9}

# Creates an OSA optimizer
o = OSA(params=params)

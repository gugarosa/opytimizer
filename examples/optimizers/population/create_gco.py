from opytimizer.optimizers.population import GCO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"CR": 0.7, "F": 1.25}

# Creates a GCO optimizer
o = GCO(params=params)

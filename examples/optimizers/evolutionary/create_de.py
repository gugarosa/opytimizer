from opytimizer.optimizers.evolutionary import DE

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {"CR": 0.9, "F": 0.7}

# Creates a DE optimizer
o = DE(params=params)

from opytimizer.optimizers.social import CI

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'r': 0.8,
    't': 3
}

# Creates an CI optimizer
o = CI(params=params)

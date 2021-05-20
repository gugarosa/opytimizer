from opytimizer.optimizers.science import TEO

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'c1': 100,
    'c2': 0.999
}

# Creates a TEO optimizer
o = TEO(params=params)

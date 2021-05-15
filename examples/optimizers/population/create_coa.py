from opytimizer.optimizers.population import COA

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'n_p': 2
}

# Creates a COA optimizer
o = COA(params=params)

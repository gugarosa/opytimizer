from opytimizer.optimizers.misc.cem import CEM

# One should declare a hyperparameters object based
# on the desired algorithm that will be used
params = {
    'n_updates': 5,
    'alpha': 0.7
}

# Creating a CEM optimizer
o = CEM(params=params)
